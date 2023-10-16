import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from keybert import KeyBERT

from create_dataset import Data_Reader, PAD
from utils import to_gpu, change_to_classify

import random
import json

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
key_bert = KeyBERT()


class MSADataset(Dataset):
    def __init__(self, config):
        dataset = Data_Reader(config)

        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode)
        self.len = len(self.data)
        prior = pickle.load(open(f'./datasets/{str.upper(config.data)}-{config.output_size}[expectation].pkl', 'rb'))
        self.prior = prior[config.mode]

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]
        config.bert_text_size = 768

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb

    def __getitem__(self, index):
        return {
            "data": self.data[index],
            "prior": self.prior,
            "index": index
        }

    def __len__(self):
        return self.len


def get_loader(config, shuffle=True, ban_word_list=[]):
    """Load DataLoader"""

    dataset = MSADataset(config)
    config.data_len = len(dataset)

    if config.dataset_name == 'split_dataset_7classes_1':
        with open('{}_train_cov_7.json'.format(config.data), 'r') as f:
            word_cov_dict = json.load(f)
        with open('{}_train_dict_7.json'.format(config.data), 'r') as f:
            word_dict = json.load(f)
    else:
        with open('{}_train_cov_2.json'.format(config.data), 'r') as f:
            word_cov_dict = json.load(f)
        with open('{}_train_dict_2.json'.format(config.data), 'r') as f:
            word_dict = json.load(f)

    if config.data == 'mosei':
        word_thresh = 825  # 825
    else:
        word_thresh = 73

    def collate_fn(batch):
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x['data'][0][0].shape[0], reverse=True)

        index = []
        for sample in batch:
            index.append(sample['index'])
        index = torch.LongTensor(index)

        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
        labels = torch.cat([torch.from_numpy(sample['data'][1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample['data'][0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample['data'][0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample['data'][0][2]) for sample in batch])
        segment = [sample['data'][2] for sample in batch]

        t_expectation = torch.cat(
            [torch.from_numpy(sample['prior']['t_expectation']).unsqueeze(0) for sample in batch], dim=0)
        a_expectation = torch.cat(
            [torch.from_numpy(sample['prior']['a_expectation']).unsqueeze(0) for sample in batch], dim=0)
        v_expectation = torch.cat(
            [torch.from_numpy(sample['prior']['v_expectation']).unsqueeze(0) for sample in batch], dim=0)

        ## BERT-based features input prep
        # SENT_LEN = sentences.size(0)

        # Create bert indices using tokenizer
        bert_details = []
        counterfactual_bert_details = []
        text_list = []

        longest_text = " ".join(batch[0]['data'][0][3])
        text_list.append(batch[0]['data'][0][3])
        longest_encoded_bert_sent = bert_tokenizer.encode_plus(longest_text, add_special_tokens=True)
        bert_details.append(longest_encoded_bert_sent)
        SENT_LEN = len(longest_encoded_bert_sent['input_ids'])
        for i in range(1, len(batch)):
            sample = batch[i]
            text = " ".join(sample['data'][0][3])
            text_list.append(sample['data'][0][3])
            encoded_bert_sent = bert_tokenizer.encode_plus(text, max_length=SENT_LEN, truncation=True,
                                                           add_special_tokens=True, padding='max_length')
            bert_details.append(encoded_bert_sent)

        for sample in batch:
            words = sample['data'][0][3]
            counterfactual_words = []
            for word in words:
                if word_cov_dict.get(word, 0) <= 0 or word_dict.get(word, 0) < word_thresh:
                    counterfactual_words.append('[MASK]')
                else:
                    counterfactual_words.append(word)
            counterfactual_text = " ".join(counterfactual_words)
            encoded_bert_sent = bert_tokenizer.encode_plus(counterfactual_text, max_length=SENT_LEN, truncation=True,
                                                           add_special_tokens=True, padding='max_length')
            counterfactual_bert_details.append(encoded_bert_sent)

        GAP_LEN = SENT_LEN - visual.size(0)
        visual_pad = torch.zeros_like(visual)[:GAP_LEN]
        acoustic_pad = torch.zeros_like(acoustic)[:GAP_LEN]
        visual = torch.cat((visual, visual_pad), dim=0)
        acoustic = torch.cat((acoustic, acoustic_pad), dim=0)

        batch_sentence_vector = []
        for sample in batch:
            sentence_vector = []
            for word in sample['data'][0][3]:
                word_id = dataset.word2id[word]
                sentence_vector.append(dataset.pretrained_emb[word_id])
            batch_sentence_vector.append(torch.stack(sentence_vector))

        counterfactual_batch_sentence_vector = []
        for sample in batch:
            counterfactual_sentence_vector = []
            for word in sample['data'][0][3]:
                if word_cov_dict.get(word, 0) <= 0 or word_dict.get(word, 0) < word_thresh:
                    word_id = dataset.word2id['<pad>']
                    counterfactual_sentence_vector.append(dataset.pretrained_emb[word_id])
                else:
                    word_id = dataset.word2id[word]
                    counterfactual_sentence_vector.append(dataset.pretrained_emb[word_id])
            counterfactual_batch_sentence_vector.append(torch.stack(counterfactual_sentence_vector))

        sentences_vector = pad_sequence(batch_sentence_vector)
        counterfactual_sentences_vector = pad_sequence(counterfactual_batch_sentence_vector)

        # Bert things are batch_first
        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
        counterfactual_bert_sentences = torch.LongTensor([sample["input_ids"] for sample in counterfactual_bert_details])
        counterfactual_bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in counterfactual_bert_details])
        counterfactual_bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in counterfactual_bert_details])

        # lengths are useful later in using RNNs
        lengths_emb = torch.LongTensor([sample['data'][0][0].shape[0] for sample in batch])
        lengths = torch.LongTensor([sample['data'][0][0].shape[0] for sample in batch])

        sample_data = {
            'index': index,

            'raw_text': text_list,
            'text': to_gpu(sentences_vector, gpu_id=config.gpu_id),  # Glove
            'counterfactual_text': to_gpu(counterfactual_sentences_vector, gpu_id=config.gpu_id),  # Glove

            'audio': to_gpu(acoustic, gpu_id=config.gpu_id),
            'visual': to_gpu(visual, gpu_id=config.gpu_id),
            'labels_classify': to_gpu(change_to_classify(labels, config), gpu_id=config.gpu_id).squeeze(),
            'labels': to_gpu(labels, gpu_id=config.gpu_id).squeeze(),
            'lengths_emb': to_gpu(lengths_emb, gpu_id=config.gpu_id),
            'lengths': to_gpu(lengths, gpu_id=config.gpu_id),
            'segment': segment,

            'bert_sentences': to_gpu(bert_sentences, gpu_id=config.gpu_id),
            'bert_sentence_att_mask': to_gpu(bert_sentence_att_mask, gpu_id=config.gpu_id),
            'bert_sentence_types': to_gpu(bert_sentence_types, gpu_id=config.gpu_id),
            'counterfactual_bert_sentences': to_gpu(counterfactual_bert_sentences, gpu_id=config.gpu_id),
            'counterfactual_bert_sentence_att_mask': to_gpu(counterfactual_bert_sentence_att_mask, gpu_id=config.gpu_id),
            'counterfactual_bert_sentence_types': to_gpu(counterfactual_bert_sentence_types, gpu_id=config.gpu_id),

            't_expectation': to_gpu(t_expectation, gpu_id=config.gpu_id),
            'a_expectation': to_gpu(a_expectation, gpu_id=config.gpu_id),
            'v_expectation': to_gpu(v_expectation, gpu_id=config.gpu_id),
        }

        return sample_data

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )

    return data_loader, len(dataset)
