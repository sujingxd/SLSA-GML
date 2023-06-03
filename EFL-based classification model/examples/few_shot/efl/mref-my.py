import logging
import os
from argparse import ArgumentParser
import json
import re
from tqdm import tqdm, trange
from pprint import pprint, pformat
import time
from datetime import timedelta
from shutil import copyfile

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import random
from model import PretrainedEmbedder, SMLP
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoModel, AdamW

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_json(d):
	with open(d, 'r', encoding='utf-8') as f:
		return json.load(f)

def write_json(data, dsave):
	outf = open(dsave, 'w', encoding='utf-8')
	json.dump(data, outf, indent=2, ensure_ascii=False)
	outf.close()
	print('>>> write to {}'.format(dsave))

def write_jsonl(data, dsave):
	outf = open(dsave, 'w')
	for d in data:
		json.dump(d, outf)
		outf.write('\n')
	outf.close()
	print('\n+++ write to {}'.format(dsave))

def custom_tokenize(text, tokenizer):
    encoded = tokenizer.encode_plus(text,
                                    add_special_tokens=True,
                                    return_token_type_ids=True)
    return encoded['input_ids']

def check_label_scores(label_scores):
    """
    avoid return of wrong argmax when multiple max values
    """
    max_v = max(label_scores)
    for i in range(3):
        if label_scores[i] == max_v:
            scores_hard = [0, 0, 0]
            scores_hard[i] = 1
            return i, scores_hard

def summary_printout(logger, content):
    for k, v in content.items():
        logger.critical('-- {}: {}'.format(k, v))
    logger.critical('\n------------------------------\n')

def format_numbers(n, decimal=4):
    d = decimal
    ns = '{:.{}f}'.format(n, d)
    return float(ns)

def format_time(seconds):
    return str(timedelta(seconds=seconds)).split('.')[0]

def save_configs(dsave, task_config):
    write_json(task_config, dsave + '/config.json')

class Trainer():
    def __init__(self, args):
        self.args = args
        self.task_config = read_json(args.config)
        ckpt_dsave = './saved/' + 'mref'

        # assert not os.path.exists(ckpt_dsave)
        # os.makedirs(ckpt_dsave)
        self.ckpt_dsave = ckpt_dsave
        print('{} created'.format(ckpt_dsave))
        self.device = torch.device('cuda:{}'.format(args.cuda))

        save_configs(ckpt_dsave, self.task_config)

        # self.model_name = 'roberta-large'
        self.model_name = 'roberta-base'
        self.lmodel = PretrainedEmbedder(self.model_name)

        # self.loaded = torch.load(ckpt_dsave+ '/model.pt.tar', map_location='cuda:{}'.format(args.cuda))  # 第一次执行可以注释掉，后期直接循环执行
        # self.lmodel.load_state_dict(self.loaded['lmodel'])

        emb_dim = self.lmodel.model.config.hidden_size
        self.layer_cls = SMLP(emb_dim, 2)

        self.vdropout = nn.Dropout(p=0.1)
        self.mse_loss = torch.nn.MSELoss()
        self.kldiv_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.vsoftmax = torch.nn.Softmax(dim=1)
        bsz =2
        self.max_seq_len = 128
        self.optimizer = None

        # bsz_test = 8
        bsz_test = 8

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.sep_token_id = tokenizer.sep_token_id

        data_dir = '3%-CR'
        train_data = self.my_load_data(data_dir+'/labeled', tokenizer, train=True, n=6)
        dev_data = self.my_load_data(data_dir+'/dev', tokenizer, train=False)
        test_data = self.my_load_data(data_dir+'/test', tokenizer, train=False)

        if args.debug:
            train_data = train_data[-100:]
            dev_data = dev_data[-100:]
            self.epochs = 3

        self.train_data = [train_data[i:i + bsz] for i in range(0, len(train_data), bsz)]
        self.dev_data = [dev_data[i:i + bsz_test] for i in range(0, len(dev_data), bsz_test)]
        self.test_data = [test_data[i:i + bsz_test] for i in range(0, len(test_data), bsz_test)]

    # def my_load_data(self, data_type, tokenizer, train=False):
    #     all_data = []
    #     data_path = r'./input_data/{}.csv'.format(data_type)
    #     dataa = pd.read_csv(data_path, sep='\t')
    #     for i in range(len(dataa)):
    #         src_id, src_polarity, src_text = dataa.loc[i][
    #             ['id', 'label', 'text', ]].values
    #         # text_raw_indices_src = tokenizer.text_to_sequence(src_text.lower().strip())
    #         text_raw_indices_src = custom_tokenize(src_text.lower().strip(), tokenizer)
    #         label = 0 if (src_polarity == '0' or src_polarity == 0) else 1
    #         # text_bert_indices_src = tokenizer.text_to_sequence('[CLS] ' + src_text.lower().strip() + ' [SEP]')
    #         bert_segments_ids_src = np.asarray([0] * (np.sum(text_raw_indices_src != 0) + 2))
    #         bert_segments_ids_src = self.pad_and_truncate(bert_segments_ids_src, 128, 'int64', 'post', 'post')
    #         text_raw_indices_src = self.pad_and_truncate(text_raw_indices_src, 128, 'int64', 'post', 'post')
    #         data = {
    #             'text_bert_indices_src': text_raw_indices_src,
    #             'bert_segments_ids_src': bert_segments_ids_src,
    #             'label': label,
    #         }
    #         all_data.append(data)
    #     if train:
    #         all_data = random.sample(all_data, len(all_data))
    #     return all_data

    def my_load_data(self, data_type, tokenizer, train=False,n=1):

        all_data = []
        data_path = r'./input_data/{}.csv'.format(data_type)
        dataa = pd.read_csv(data_path, sep='\t')
        for i in range(len(dataa)):
            src_polarity, src_text_ori = dataa.loc[i][
                ['label', 'text', ]].values
            src_text = src_text_ori.strip().lower()
            input_ids = custom_tokenize(src_text, tokenizer)
            input_mask = [1] * len(input_ids)
            input_ids = input_ids[:self.max_seq_len]
            input_mask = input_mask[:self.max_seq_len]
            while len(input_ids) < self.max_seq_len:
                input_ids.append(1)
                input_mask.append(0)

            text_raw_indices_src = np.asarray(input_ids, dtype='float')
            input_mask = np.asarray(input_mask, dtype='float')

            label = 0 if (src_polarity == '0' or src_polarity == 0) else 1

            N_SAMPLES = n # 对于每一个数据生成6个例子
            for i in range(N_SAMPLES):
                if N_SAMPLES > 1:
                    masked_text_raw_indices_src = self.replace_with_mask_tokens(text_raw_indices_src, p=0.25)
                else:
                    masked_text_raw_indices_src = text_raw_indices_src
                data = {
                    'text_bert_indices_src': masked_text_raw_indices_src,
                    'bert_segments_ids_src': input_mask,
                    'text':src_text_ori,
                    'label': label,
                    'label_scores':[0,1] if label==1 else [1,0],
                    'label_gold' :label,
                    'label_scores_hard':[0,1] if label==1 else [1,0],
                }
                all_data.append(data)
        if train:
            all_data = random.sample(all_data, len(all_data))
        return all_data


    def pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x


    # def load_data(self, data, tokenizer, train=False, n=1):
    #     print('... loading data ... train={}'.format(train))
    #     datap = []
    #
    #     for inst in data:
    #         uid = inst['uid']
    #         prev_turns = inst['prev_turns']
    #         label_scores = inst['label_scores']
    #         label_scores_hard = label_scores
    #         label_gold = 0
    #         if label_scores == [0, 1]:
    #             label_gold = 1
    #
    #         metadata = {}
    #         metadata['uid'] = uid
    #         metadata['context_id'] = inst['context_id']
    #         metadata['prev_turns'] = prev_turns
    #         metadata['utterance'] = inst['utterance']
    #
    #
    #         context_turns_selected = inst['context']
    #         c_text = ' '.join(context_turns_selected)
    #         t_text = inst['utterance']
    #
    #         prev_turns_selected = prev_turns
    #         h_text = ' '.join(prev_turns_selected)
    #
    #         combined_text = c_text + ' ' + h_text + ' ' + tokenizer.sep_token + ' ' + t_text
    #
    #         max_seq_len = 256
    #
    #         while len(combined_text.split()) > max_seq_len:
    #             if len(context_turns_selected) > 0:
    #                 context_turns_selected = context_turns_selected[:-1]
    #             else:
    #                 prev_turns_selected = prev_turns_selected[1:]
    #             c_text = ' '.join(context_turns_selected)
    #             h_text = ' '.join(prev_turns_selected)
    #             combined_text = c_text + ' ' + h_text + ' ' + tokenizer.sep_token + ' ' + t_text
    #         combined_tokens = custom_tokenize(combined_text, tokenizer)
    #
    #         while len(combined_tokens) > max_seq_len:
    #             if len(context_turns_selected) > 0:
    #                 context_turns_selected = context_turns_selected[:-1]
    #             else:
    #                 prev_turns_selected = prev_turns_selected[1:]
    #             c_text = ' '.join(context_turns_selected)
    #             h_text = ' '.join(prev_turns_selected)
    #             combined_text = c_text + ' ' + h_text + ' ' + tokenizer.sep_token + ' ' + t_text
    #             combined_tokens = custom_tokenize(combined_text, tokenizer)
    #
    #         N_SAMPLES = n
    #         for i in range(N_SAMPLES):
    #             if N_SAMPLES > 1:
    #                 masked_combined_tokens = self.replace_with_mask_tokens(combined_tokens, p=0.25)
    #             else:
    #                 masked_combined_tokens = combined_tokens
    #
    #             datap_inst = {'combined_tokens': masked_combined_tokens,
    #                           'label_scores': label_scores,
    #                           'label_scores_hard': label_scores_hard,
    #                           'label_gold': label_gold,
    #                           'metadata': metadata
    #                           }
    #             datap.append(datap_inst)
    #     if train:
    #         datap = random.sample(datap, len(datap))
    #     return datap



    def replace_with_mask_tokens(self, seq, p=0):
        if len(seq) <= 2:
            return seq
        seq = torch.LongTensor(seq)
        seq_mask = torch.rand(seq.shape[0]-2).lt(p)
        seq[1:-1][seq_mask==True] = self.tokenizer.mask_token_id
        seq = seq.tolist()
        return seq

    def output_test_results(self, test_data):
        test_samples = [d['label_probs'] for d in test_data]
        test_labels = [d['label_gold'] for d in test_data]
        test_preds = [1 if score > 0.5 else 0 for score in test_samples]
        report = classification_report(test_labels, test_preds,  digits=3, labels=[0, 1],output_dict=True)
        V_P = report['1']['precision']
        V_R = report['1']['recall']
        V_f1 = report['1']['f1-score']
        macro_f1 = report['macro avg']['f1-score']

        V_P = '{:.1f}'.format(V_P * 100)
        V_R = '{:.1f}'.format(V_R * 100)
        V_f1 = '{:.1f}'.format(V_f1 * 100)
        macro_f1 = '{:.1f}'.format(macro_f1 * 100)

        return V_P, V_R, V_f1, macro_f1

    def log_test_results(self, test_data, test_name):
        V_P, V_R, V_f1, macro_f1 = self.output_test_results(test_data)
        self.logger.critical('{:<9}  {} / {} / {} / {}'.format(test_name, V_P, V_R, V_f1, macro_f1))

    def run_process(self):  # 整体的全局运行
        global_start_time = time.time()
        ckpt_dsave = self.ckpt_dsave
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(ckpt_dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)
        self.logger = logger
        logger.critical('------ task config ------')
        # logger.critical('-- cuda: {}'.format(self.args.cuda))
        summary_printout(logger, self.task_config)

        self.lmodel = self.lmodel.to(self.device)
        self.layer_cls = self.layer_cls.to(self.device)

        param_groups = [{'params': self.lmodel.parameters()},
                        {'params': self.layer_cls.parameters()}]

        lr = 2e-6#2e-6

        epochs = 10#5
        patience_max = 5  #5

        self.optimizer = AdamW(params=param_groups, lr=lr, weight_decay=0.01)

        ckpt_best = None
        dev_acc_best = 0.0
        epoch_best = 0
        patience = 0

        summary = []

        epoch_i = 0

        for epoch in trange(epochs, desc='epoch'):
            epoch_start_time = time.time()

            _, train_acc, loss_terms = self.run_subprocess(self.train_data, train=True)  # 部分的运行

            dev_out, dev_acc, valid_loss_terms = self.run_subprocess(self.dev_data, dev=True) # 部分的运行
            epoch_i += 1

            V_P, V_R, V_f1, macro_f1 = self.output_test_results(dev_out)
            dev_acc = float(V_f1)

            logger.critical('----- epoch {} -----'.format(epoch))

            if dev_acc > dev_acc_best:
                dev_acc_best = dev_acc
                epoch_best = epoch
                ckpt_best = {'epoch': epoch_best,
                             'lmodel': self.keep_state_dict(self.lmodel.cpu().state_dict()),
                             'layer_cls': self.keep_state_dict(self.layer_cls.cpu().state_dict()),
                             'train_acc': train_acc,
                             'dev_acc': dev_acc_best
                             }
                self.lmodel = self.lmodel.to(self.device)
                self.layer_cls = self.layer_cls.to(self.device)

                patience = 0
            else:
                patience += 1

            summary.append({'epoch': epoch,
                            'epoch_best': epoch_best,
                            'train_acc': train_acc,
                            'dev_acc': dev_acc,
                            'epoch_time': format_time(time.time() - epoch_start_time)
                            })
            logger.critical('\n\n------ summary: epoch {} ----\n'.format(epoch))
            logger.critical('-- train loss ---> {}'.format(loss_terms))
            logger.critical('-- valid loss ---> {}'.format(valid_loss_terms))
            summary_printout(logger, summary[-1])

            if patience == patience_max or epoch == epochs - 1:
                torch.save(ckpt_best, ckpt_dsave + '/model.pt.tar')
                if patience == patience_max:
                    logger.critical('best epoch: {}. patience {} reached.'.format(epoch_best, patience_max))
                else:
                    logger.critical('best epoch: {}.'.format(epoch_best))
                logger.critical('------ training summary ------')
                summary_printout(logger, summary[epoch_best])
                logger.critical('total epochs: {}'.format(epoch))
                logger.critical('total time: {}'.format(format_time(time.time() - global_start_time)))
                logger.critical('model directory: {}'.format(ckpt_dsave))
                logger.critical('------------------------------')
                logger.critical('best ckpt saved: model.pt.tar'.format(epoch_best))
                break
        test_out, test_acc = self.run_subprocess(self.test_data, test=True)  # 部分的运行
        train_out, _ = self.run_subprocess(self.train_data, test=True)  # 部分的运行
        dev_out, _ = self.run_subprocess(self.dev_data, test=True)  # 部分的运行
        print('test_acc is: '+ str(test_acc))
        res_pdf=pd.DataFrame()
        new_ids = []
        new_texts = []
        new_label_preds= []
        new_true_label = []
        for ele in test_out:
            new_ids.append(ele['id'])
            new_texts.append(ele['text'])
            new_label_preds.append(ele['label_probs'])
            new_true_label.append(ele['label_gold'])
        res_pdf['id'] = new_ids
        res_pdf['text'] = new_texts
        res_pdf['label_probs'] = new_label_preds
        res_pdf['label_gold'] = new_true_label
        res_pdf.to_csv('mref_test_result.csv',index=None,sep='\t')


        train_pdf=pd.DataFrame()
        new_ids = []
        new_texts = []
        new_label_preds= []
        new_true_label = []
        for ele in train_out:
            new_ids.append(ele['id'])
            new_texts.append(ele['text'])
            new_label_preds.append(ele['label_probs'])
            new_true_label.append(ele['label_gold'])
        train_pdf['id'] = new_ids
        train_pdf['text'] = new_texts
        train_pdf['label_probs'] = new_label_preds
        train_pdf['label_gold'] = new_true_label
        train_pdf.to_csv('mref_train_result.csv',index=None,sep='\t')

        dev_pdf=pd.DataFrame()
        new_ids = []
        new_texts = []
        new_label_preds= []
        new_true_label = []
        for ele in dev_out:
            new_ids.append(ele['id'])
            new_texts.append(ele['text'])
            new_label_preds.append(ele['label_probs'])
            new_true_label.append(ele['label_gold'])
        dev_pdf['id'] = new_ids
        dev_pdf['text'] = new_texts
        dev_pdf['label_probs'] = new_label_preds
        dev_pdf['label_gold'] = new_true_label
        dev_pdf.to_csv('mref_dev_result.csv',index=None,sep='\t')

    def keep_state_dict(self, state_dict):
        import copy
        return copy.deepcopy(state_dict)

    def soft_frequency(self, logits,  probs=False, soft = True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = 1  # power=1
        if not probs: # not enter
            softmax = nn.Softmax(dim=1)
            y = softmax(logits.view(-1, logits.shape[-1])).view(logits.shape)
        else:
            y = logits
        f = torch.sum(y, dim=0)
        t = y**power / f
        t = t + 1e-10
        p = t/torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)

    def get_knn(self, current_id, k, list, label,train_len,type):
        list = list.tolist()
        list_sorted = sorted(enumerate(list), key=lambda x:x[1])
        sorted_indexs = [m[0] for m in list_sorted]
        sorted_indexs.remove(current_id)
        ids = sorted_indexs[:k]
        knn_labels = [label[id] for id in ids]
        knn_distance = [list[id] for id in ids]
        return knn_labels, ids, knn_distance

    def run_subprocess(self, data, train=False, dev=False, test=False, gen=False, unlabeled=False):
        if train or unlabeled:
            self.lmodel.train()
            self.layer_cls.train()
        else:
            self.lmodel.eval()
            self.layer_cls.eval()

        device = self.device
        train_loss = 0.0
        valid_loss_CE = 0.0
        run_correct = 0
        seen = 0
        inst_count = 0
        loss_terms = {}

        sum_CE = 0.0
        sum_SCL = 0.0

        acc = 0.0
        pred_out = []
        val_iteration = len(data) if (len(data) <2000) else 2000  # 这个随意取得，只要大于等于原始数据集应该就可以，
        if train == False:
            val_iteration = len(data)
            self.dataiter = iter(data)
        from tqdm import tqdm, trange
        iters = trange(val_iteration)
        for batch_idx in iters:
            if train:
                try:
                    batch = next(self.trainiter)
                except:
                    self.trainiter = iter(data)
                    batch = next(self.trainiter)
            else:
                batch = next(self.dataiter)

            if train or unlabeled:
                self.optimizer.zero_grad()
            # combined_tokens_b = pad_sequence([torch.LongTensor(d['combined_tokens']) for d in batch], batch_first=True,
            #                                  padding_value=self.pad_token_id).to(device)
            # attn_mask_b = pad_sequence([torch.FloatTensor([1]*len(d['combined_tokens'])) for d in batch], batch_first=True,
            #                             padding_value=0).to(device)

            # combined_tokens_b = ([self.pad_and_truncate(d['text_bert_indices_src'], 128, 'int64', 'post', 'post') for d in batch])
            # attn_mask_b = ([self.pad_and_truncate(d['text_bert_indices_src'], 128, 'int64', 'post', 'post') for d in batch])
            # combined_tokens_b = pad_sequence([torch.LongTensor(d) for d in combined_tokens_b], batch_first=True, padding_value=self.pad_token_id).to(device)
            # attn_mask_b = pad_sequence([torch.FloatTensor([0]*len(d)) for d in attn_mask_b], batch_first=True,
            #                             padding_value=0).to(device)


            combined_tokens_b = ([self.pad_and_truncate(d['text_bert_indices_src'], 128, 'int64', 'post', 'post') for d in batch])

            combined_tokens_b = pad_sequence([torch.LongTensor(d) for d in combined_tokens_b], batch_first=True,
                                             padding_value=self.pad_token_id).to(device)

            attn_mask = pad_sequence([torch.LongTensor(d['bert_segments_ids_src']) for d in batch], batch_first=True,
                                             padding_value=self.pad_token_id).to(device)

            if not (unlabeled or gen):
                label_scores = torch.FloatTensor([d['label_scores'] for d in batch]).to(device)
                label_scores_hard = torch.FloatTensor([d['label_scores_hard'] for d in batch])
                labels = label_scores_hard.argmax(dim=1).to(device)
                label_gold = [d['label_gold'] for d in batch]

            if not train:
                with torch.no_grad():
                    combined_out = self.lmodel(combined_tokens_b, attn_mask)[0][:, 0, :]
                    x = self.layer_cls(combined_out)
                    if gen:
                        gen_tau = 1.0
                        output = self.vsoftmax(torch.div(x, gen_tau))
                    else:
                        output = self.vsoftmax(x)
                    _, pred_label = output.max(1)

            if train:
                combined_out = self.lmodel(combined_tokens_b, attn_mask)[0][:, 0, :]
                x = self.layer_cls(combined_out)

                output = self.vsoftmax(x)
                L_SCL = self.SCL_loss(F.normalize(combined_out, dim=1), labels)

            if unlabeled:
                pass


            seen += len(batch)
            ## sjj
            import sklearn.metrics.pairwise as pairwise
            pairwise_distances = pairwise.pairwise_distances(combined_out.detach().cpu().numpy(), combined_out.detach().cpu().numpy(), metric="cosine")
            knn_train = []
            knn_ids_train = []
            knn_distance_train = []
            weight = []
            soft = True
            softmax = nn.Softmax(dim=1)
            temp_target = softmax(x.view(-1, x.shape[-1])).view(x.shape)
            arg_max_logits = torch.argmax(self.soft_frequency(temp_target, probs = True, soft = soft),dim=1)
            from tqdm import trange, tqdm
            k = 4
            for i in trange(len(pairwise_distances)):
                temp = self.get_knn(i, k, pairwise_distances[i], arg_max_logits, pairwise_distances.shape[0], 'train')
                knn_train.append(temp[0])
                knn_ids_train.append(temp[1])
                knn_distance_train.append(temp[2])
                label_of_this_sample = arg_max_logits[i]
                count = 0
                for each_i in  temp[0]:
                    if each_i.item() ==label_of_this_sample.item():
                        count +=1
                import math
                weight.append(count/(k))
            sorted_list = sorted(weight)
            max_value = 0.8
            new_weight = []
            for ind, t in enumerate(weight):
                if t == 1:
                    new_weight.append(t)
                else:
                    new_weight.append(t/max_value)
            softmax = nn.Softmax(dim=1)
            target = softmax(output.view(-1, output.shape[-1])).view(output.shape)

            ten_weight = torch.tensor(new_weight,device='cuda').reshape(len(pairwise_distances),1)
            weight = torch.sum(-torch.log(ten_weight+1e-6) * ten_weight, dim = 1)
            weight = 1 - weight / np.log(weight.size(-1))
            thresh = 0.3
            w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(self.device)
            target = self.soft_frequency(target, probs = True, soft = soft)
            loss_batch = self.cross_entropy(x, torch.tensor(labels).to(self.device))
            L_CE = torch.sum(loss_batch * w.unsqueeze(1) * weight.unsqueeze(1))


            ## sjj
            if train:
                p1 = 1.0
                # p2 = 0.1
                # p3 = 0.5
                p2 = 0.1
                # L_CE = self.cross_entropy(x, labels.long().cuda())  # 交叉熵

                sum_CE += L_CE.item() * len(batch)
                sum_SCL += L_SCL.item() * len(batch)

                avg_CE = sum_CE / seen
                avg_SCL = sum_SCL / seen

                # loss = p1 * L_CE + p2 * L_SCL
                loss = p1 * L_CE + p2 * L_SCL

                loss.backward()
                train_loss += (loss.item() * len(batch))      # criterion default reduction = 'mean'
                avg_loss = train_loss / seen
                self.optimizer.step()

                loss_terms = {'CE': format_numbers(avg_CE),
                              'SCL': format_numbers(avg_SCL),
                              'AggAvg': format_numbers(avg_loss)}

            if unlabeled: # false
                loss = L_SCL
                sum_SCL += L_SCL.item() * len(batch)
                avg_SCL = sum_SCL / seen

                loss.backward()

                train_loss += (loss.item() * len(batch))

                avg_loss = train_loss / seen
                self.optimizer.step()
                loss_terms = {'SCL': format_numbers(avg_SCL),
                              'AggAvg': format_numbers(avg_loss)}

            if dev: # fasle
                with torch.no_grad():
                    L_CE = self.cross_entropy(x, labels.long().cuda())

                    valid_loss_CE += L_CE.item() * len(batch)

                    valid_loss_CE_avg = valid_loss_CE / seen

                    loss_terms = {'CE': format_numbers(valid_loss_CE_avg)}


            if not (unlabeled or gen):
                run_correct += (output.detach().cpu().argmax(dim=1) == label_scores_hard.argmax(dim=1)).sum().item()
                acc = run_correct/seen


            # update tqdm bar display

            if train:
                iters.set_description('CE:{:.3f}|SCL:{:.3f}|Acc:{:.4f}'.
                                          format(avg_CE, avg_SCL, acc), refresh=False)
            if unlabeled:
                iters.set_description('SCL:{:.3f}'.format(avg_SCL), refresh=False)
            if dev:
                iters.set_description('[dev] acc_cls:{:.4f}'.format(acc), refresh=False)


            if dev or test:
                assert len(batch) == output.shape[0] == pred_label.shape[0] == \
                       label_scores.shape[0] == len(label_gold)
                for d, dist, cls, lp_gold, lgold_hard in zip(batch, output, pred_label, label_scores, label_gold):
                    pred_out.append({'id': 'eval_' + str(inst_count),
                                     # 'uid': d['metadata']['uid'],
                                     # 'context_id': d['metadata']['context_id'],
                                     'text': d['text'],
                                     'label_p_gold': lp_gold.tolist(),
                                     'label_gold': lgold_hard,
                                     'label_probs': dist.tolist()[1],
                                     'label_pred': int(cls)
                                     })

                    inst_count += 1

        if train or dev or unlabeled:
            return pred_out, acc, loss_terms
        else:
            return pred_out, acc

    def SCL_loss(self, features, labels, tau=1.0, tau_base=1.0):
        labels = labels.contiguous().view(-1, 1)
        mask_s = torch.eq(labels, labels.T).float().fill_diagonal_(0)
        mask_s = mask_s.to(self.device)

        sims = torch.div(features.mm(features.T), tau)
        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()

        logits_mask = torch.ones_like(mask_s).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        denom = mask_s.sum(1)
        denom[denom==0] = 1

        mean_log_prob_pos = (mask_s * log_prob).sum(1) / denom

        loss = - (tau/tau_base) * mean_log_prob_pos
        loss = loss.mean()

        return loss

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run_process()