import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from models.misa_model import MISA
from models.selfmm_model import SELF_MM
from models.magbert_model import BERT_MAG
import torch.nn.functional as F

from models.subNets.BertTextEncoder import BertTextEncoder
from transformers import BertTokenizer, BertModel


class Causal_Model(nn.Module):
    def __init__(self, config):
        super(Causal_Model, self).__init__()

        self.config = config
        self.fusion_mode = config.fusion_mode
        self.output_size = config.output_size
        self.hidden_size = config.tmodel_hidden_size
        self.embedding_size = config.tmodel_embedding_size
        self.m_heads = config.m_heads

        # loss function
        self.classify_criterion = nn.CrossEntropyLoss()

        # multimodal -- basemodel
        if self.config.base_model == "misa_model":
            self.base_model = MISA(config)
            m_dim = 384
        elif self.config.base_model == "selfmm_model":
            self.base_model = SELF_MM(config)
            m_dim = 128
        elif self.config.base_model == "magbert_model":
            self.base_model = BERT_MAG(config)
            m_dim = 768
        else:
            raise NameError('No {} model can be found'.format(self.config.base_model))
        self.m_dim = m_dim

        # multimodal -- text model
        if self.config.use_bert:
            self.bert = BertEncoder(language='en', use_finetune=True)
        self.text_model = nn.GRU(input_size=config.text_size, hidden_size=self.embedding_size)
        self.text_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),
        )
        self.text_classifier = nn.Linear(self.hidden_size, self.output_size)

        # multimodal -- audio model
        self.audio_model = nn.GRU(input_size=config.audio_size, hidden_size=self.embedding_size)
        self.audio_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),
        )
        self.audio_classifier = nn.Linear(self.hidden_size, self.output_size)

        # multimodal -- video model
        self.video_model = nn.GRU(input_size=config.video_size, hidden_size=self.embedding_size)
        self.video_mlp = nn.Sequential(
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_size)
        )
        self.video_classifier = nn.Linear(self.hidden_size, self.output_size)

        # intervention
        if self.config.use_intervention:
            self.m_proj = nn.Linear(self.m_dim, self.m_dim * self.m_heads)
            self.x_proj_q = nn.Linear(768 + config.audio_size + config.video_size, self.hidden_size)
            self.m_proj_k = nn.Linear(self.m_dim, self.hidden_size)
            self.scale = self.hidden_size ** -0.5
            self.m_classifier = nn.Linear(
                768 + config.audio_size + config.video_size + self.m_dim, self.output_size)

        if self.config.fusion_mode == 'fc':
            self.fc = nn.Linear(self.output_size * 4, self.output_size)

        self.constant = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch_sample, labels, epoch_info={'batch_index': 0, 'epoch_index': -1}):
        cf_sentences = batch_sample['counterfactual_text']
        audio = batch_sample['audio']  # for mosi: 74, for mosei: 74
        video = batch_sample['visual']  # for mosi: 47, for mosei: 35
        cf_input_ids = batch_sample['counterfactual_bert_sentences']
        cf_input_mask = batch_sample['counterfactual_bert_sentence_att_mask']
        cf_segment_ids = batch_sample['counterfactual_bert_sentence_types']
        t_expectation = batch_sample['t_expectation']
        a_expectation = batch_sample['a_expectation']
        v_expectation = batch_sample['v_expectation']
        lengths = batch_sample['lengths']
        bs = lengths.size(0)

        # basemodel output
        base_output = self.base_model(batch_sample, epoch_info)
        o_multimodal = base_output['o_multimodal']
        m_rep = base_output['m_rep']
        base_loss = base_output['loss']

        x_t = base_output['t_rep']
        x_a = torch.mean(audio, dim=0)
        x_v = torch.mean(video, dim=0)

        # textmodel output
        if self.config.use_bert:
            text = self.bert(cf_input_ids, cf_input_mask, cf_segment_ids).transpose(0, 1)
        else:
            text = cf_sentences
        _, t_rep = self.text_model(text)
        t_rep = self.text_mlp(t_rep.squeeze())

        # audiomodel output
        _, a_rep = self.audio_model(audio)
        a_rep = self.audio_mlp(a_rep.squeeze())

        # videomodel output
        _, v_rep = self.video_model(video)
        v_rep = self.video_mlp(v_rep.squeeze())

        if self.config.use_intervention:
            mh_m = F.relu(self.m_proj(m_rep)).view(bs, self.m_heads, self.m_dim)
            m_k = F.relu(self.m_proj_k(mh_m))
            x_q = F.relu(self.x_proj_q(torch.cat((x_t, x_a, x_v), dim=-1))).unsqueeze(1) * self.scale
            m_x_attn = F.softmax(torch.bmm(x_q, m_k.transpose(1, 2)), dim=-1)
            m_x = torch.bmm(m_x_attn, mh_m).squeeze()
            o_multimodal = self.m_classifier(
                torch.cat((t_expectation, a_expectation, v_expectation, m_x), dim=-1))

        o_multimodal_c = self.constant * torch.ones_like(o_multimodal).cuda(self.config.gpu_id)
        o_text = self.text_classifier(t_rep)
        o_audio = self.audio_classifier(a_rep)
        o_video = self.video_classifier(v_rep)

        o1_fusion = self.fusion_function(o_multimodal, o_text, o_audio, o_video)
        o2_fusion = self.fusion_function(o_multimodal_c, o_text, o_audio, o_video)

        output = {
            'o_multimodal': o_multimodal,
            'o_multimodal_c': o_multimodal_c,
            'o_text': o_text,
            'o_audio': o_audio,
            'o_video': o_video,
            'o1_fusion': o1_fusion,
            'o2_fusion': o2_fusion,
            'x_t': x_t,
        }

        labels = labels.long()
        output['labels'] = labels
        output['base_loss'] = base_loss

        if self.config.only_base_model:
            cls_loss = self.classify_criterion(o_multimodal, labels)
            output['cls_loss'] = cls_loss
        else:
            cls_loss = self.classify_criterion(o1_fusion, labels) + \
                            self.classify_criterion(o_text, labels) + \
                            self.classify_criterion(o_audio, labels) + \
                            self.classify_criterion(o_video, labels)
            output['cls_loss'] = cls_loss
        output['loss'] = base_loss + cls_loss

        return output

    def fusion_function(self, o1, o2, o3, o4, use_kl=False):
        if self.fusion_mode == "sum":
            o_fusion = F.logsigmoid(o1 + o2 + o3 + o4)

        elif self.fusion_mode == "hm":
            o = torch.sigmoid(o1) * torch.sigmoid(o2) * torch.sigmoid(o3) * torch.sigmoid(o4)
            o_fusion = torch.log(o / (1 + o))

        elif self.fusion_mode == 'rubi':
            o_fusion = o1 * torch.sigmoid(o2 + o3 + o4)

        elif self.fusion_mode == 'fc':
            if use_kl:
                with torch.no_grad():
                    o_fusion = self.fc(torch.cat((o1, o2, o3, o4), dim=-1))  # not upgrade fusion parameters during kl_loss
            else:
                o_fusion = self.fc(torch.cat((o1, o2, o3, o4), dim=-1))

        return o_fusion


class BertEncoder(nn.Module):
    def __init__(self, language='en', use_finetune=False):
        """
        language: en / cn
        """
        super(BertEncoder, self).__init__()

        assert language in ['en', 'cn']

        tokenizer_class = BertTokenizer
        model_class = BertModel
        if language == 'en':
            self.tokenizer = tokenizer_class.from_pretrained('bert-base-uncased', do_lower_case=True)
            self.model = model_class.from_pretrained('bert-base-uncased')
        elif language == 'cn':
            self.tokenizer = tokenizer_class.from_pretrained('pretrained_bert/bert_cn')
            self.model = model_class.from_pretrained('pretrained_bert/bert_cn')

        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def from_text(self, text):
        """
        text: raw data
        """
        input_ids = self.get_id(text)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]  # Models outputs are now tuples
        return last_hidden_states.squeeze()

    def forward(self, input_ids, input_mask, segment_ids):
        """
        input_ids: input_ids (batch_size, seq_len),
        input_mask: attention_mask (batch_size, seq_len),
        segment_ids: token_type_ids (batch_size, seq_len)
        """
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]  # Models outputs are now tuples
        return last_hidden_states
