# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import random
import time
import json
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset

from data import create_dataloader, convert_example, processor_dict
from task_label_description import TASK_LABELS_DESC


# yapf: disable
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, type=str, help="The task_name to be evaluated")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--output_dir", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument("--seed", type=int, default=1000, help="random seed for initialization")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument('--save_steps', type=int, default=10000, help="Inteval steps to save checkpoint")

    return parser.parse_args()

# yapf: enable


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def get_data(distribution_train):
    new_distribution_train = []
    for i in range(len(distribution_train)):
        if i %2 ==0:
            new_distribution_train.append(distribution_train[i])
    return np.array(new_distribution_train)

def get_data_1(distribution_train):
    new_distribution_train = []
    for i in range(len(distribution_train)):
        if i %2 ==1:
            new_distribution_train.append(distribution_train[i])
    return np.array(new_distribution_train)


@paddle.no_grad()
def do_predict(model, tokenizer, data_loader, data_type, data_name, task_label_description):
    model.eval()

    index2label = {
        idx: label
        for idx, label in enumerate(task_label_description.keys())
    }

    class_num = len(task_label_description)

    all_prediction_probs = []
    t_sentence_out = None
    t_pooled_output_skepmodel = None
    t_content = None
    for batch in data_loader:
        if data_type == 'test' :
            src_ids, token_type_ids = batch
        else:
            src_ids, token_type_ids,_= batch

        # Prediction_probs:[bs, 2]
        prediction_probs, content, pooled_output_skepmodel,sentence_out = model(
            input_ids=src_ids, token_type_ids=token_type_ids)
        # prediction_probs  = model(
        #     input_ids=src_ids, token_type_ids=token_type_ids)
        max_len = 128
        if t_sentence_out is None:
            t_sentence_out = sentence_out.detach().cpu().numpy()
            t_pooled_output_skepmodel = pooled_output_skepmodel.detach().cpu().numpy()
            content = content.detach().cpu().numpy()
            btz = content.shape[0]
            cur_len= content.shape[1]
            dim = content.shape[2]
            if cur_len <= 128:
                pad_np = np.zeros((btz, max_len-cur_len,dim))
                t_content  = np.concatenate((content,pad_np),axis=1)
            else :
                t_content = content[:,:128,:]
        else:
            t_sentence_out = np.concatenate((t_sentence_out, sentence_out.detach().cpu().numpy()),axis=0)
            t_pooled_output_skepmodel = np.concatenate((t_pooled_output_skepmodel, pooled_output_skepmodel.detach().cpu().numpy()),axis=0)
            content = content.detach().cpu().numpy()
            btz = content.shape[0]
            cur_len= content.shape[1]
            dim = content.shape[2]
            if cur_len <= 128:
                pad_np = np.zeros((btz, max_len-cur_len,dim))
                new_content  = np.concatenate((content, pad_np),axis=1)
            else:
                new_content = content[:,:128,:]
            t_content = np.concatenate((t_content, new_content),axis=0)
        all_prediction_probs.append(prediction_probs.numpy())

    t_sentence_out0 = get_data(t_sentence_out)
    t_sentence_out1 = get_data_1(t_sentence_out)
    t_pooled_output_skepmodel1 = get_data_1(t_pooled_output_skepmodel)
    t_content = get_data_1(t_content)
    if data_type == 'test' and data_name=='no':
        model.train()
    else:
        sentence_out_file0 = '{}_{}_sentence_out0.csv'.format(data_type, data_name)
        np.savetxt(sentence_out_file0, t_sentence_out0, fmt="%.17f",  delimiter="\t")
        sentence_out_file1 = '{}_{}_sentence_out1.csv'.format(data_type, data_name)
        np.savetxt(sentence_out_file1, t_sentence_out1, fmt="%.17f",  delimiter="\t")
        t_pooled_output_skepmodel1_file1 = '{}_{}_output_skepmode1.csv'.format(data_type, data_name)
        np.savetxt(t_pooled_output_skepmodel1_file1, t_pooled_output_skepmodel1, fmt="%.17f",  delimiter="\t")
        t_content_file1 = '{}_{}_content_skepmodel_out1.pkl'.format(data_type, data_name)
        t_content_file1_pk = open(t_content_file1, 'wb')
        # np.savetxt(t_content_file1, t_content, fmt="%.17f",  delimiter="\t")
        import pickle
        pickle.dump(t_content_file1, t_content_file1_pk, protocol = 4)

        # pass

    all_prediction_probs = np.concatenate(all_prediction_probs, axis=0)

    all_prediction_probs = np.reshape(all_prediction_probs, (-1, class_num, 2))

    prediction_pos_probs = all_prediction_probs[:, :, 1]
    prediction_pos_probs = np.reshape(prediction_pos_probs, (-1, class_num))
    y_pred_index = np.argmax(prediction_pos_probs, axis=-1)

    y_preds = [index2label[idx] for idx in y_pred_index]

    model.train()
    prediction_pos_probs = paddle.to_tensor(prediction_pos_probs)
    prediction_pos_probs = F.softmax(prediction_pos_probs,  1)
    _pos_prob_list = list()
    for (neg_prob, pos_prob) in prediction_pos_probs:
        pos_prob = pos_prob.item()
        _pos_prob_list.append(pos_prob)
    return y_preds, _pos_prob_list


def write_iflytek(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = str(pred_labels[idx])

            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


def write_bustm(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


def write_csldcp(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                "id", test_example['id'], "label", test_example["label"])
            f.write("{" + str_test_example + "}\n")


def write_tnews(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]

            str_test_example = json.dumps(test_example) + "\n"
            f.write(str_test_example)


def write_cluewsc(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = "\"{}\": {}, \"{}\": \"{}\"".format(
                "id", test_example['id'], "label", test_example["label"])
            f.write("{" + str_test_example + "}\n")


# def write_eprstmt(task_name, output_file, pred_labels):
#     test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
#     test_example = {}
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for idx, example in enumerate(test_ds):
#             test_example["id"] = example["id"]
#             test_example["label"] = pred_labels[idx]
#
#             str_test_example = json.dumps(test_example)
#             f.write(str_test_example + "\n")


def write_eprstmt(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(pred_labels):
            test_example["id"] = idx
            test_example["label"] = int(pred_labels[idx])

            str_test_example = json.dumps(test_example)
            f.write(str_test_example + "\n")
            
            
def write_ocnli(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = json.dumps(test_example)
            f.write(str_test_example + "\n")


def write_csl(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["label"] = pred_labels[idx]
            str_test_example = json.dumps(test_example)
            f.write(str_test_example + "\n")


def write_chid(task_name, output_file, pred_labels):
    test_ds = load_dataset("fewclue", name=task_name, splits=("test"))
    test_example = {}
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_ds):
            test_example["id"] = example["id"]
            test_example["answer"] = pred_labels[idx]
            str_test_example = "\"{}\": {}, \"{}\": {}".format(
                "id", test_example['id'], "answer", test_example["answer"])
            f.write("{" + str_test_example + "}\n")


predict_file = {
    "bustm": "bustm_predict.json",
    "chid": "chidf_predict.json",
    "cluewsc": "cluewscf_predict.json",
    "csldcp": "csldcp_predict.json",
    "csl": "cslf_predict.json",
    "eprstmt": "eprstmt_predict.json",
    "iflytek": "iflytekf_predict.json",
    "ocnli": "ocnlif_predict.json",
    "tnews": "tnewsf_predict.json"
}

write_fn = {
    "bustm": write_bustm,
    "iflytek": write_iflytek,
    "csldcp": write_csldcp,
    "tnews": write_tnews,
    "cluewsc": write_cluewsc,
    "eprstmt": write_eprstmt,
    "ocnli": write_ocnli,
    "csl": write_csl,
    "chid": write_chid
}

if __name__ == "__main__":
    args = parse_args()
    paddle.set_device(args.device)
    set_seed(args.seed)

    processor = processor_dict[args.task_name]()
    # Load test_ds for FewCLUE leaderboard
    test_ds = load_dataset("fewclue", name=args.task_name, splits=("test"))
    test_ds = processor.get_test_datasets(test_ds,
                                          TASK_LABELS_DESC[args.task_name])

    model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
        'ernie-1.0', num_classes=2)
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    # [src_ids, token_type_ids]
    predict_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    ): [data for data in fn(samples)]

    predict_trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    test_data_loader = create_dataloader(
        test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=predict_batchify_fn,
        trans_fn=predict_trans_func)

    # Load parameters of best model on test_public.json of current task
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.init_from_ckpt)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    y_pred_labels = do_predict(
        model,
        tokenizer,
        test_data_loader,
        task_label_description=TASK_LABELS_DESC[args.task_name])
    output_file = os.path.join(args.output_dir, predict_file[args.task_name])
    write_fn[args.task_name](args.task_name, output_file, y_pred_labels)
