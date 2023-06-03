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
from paddlenlp.transformers import LinearDecayWithWarmup

from data import create_dataloader, convert_example, processor_dict
from evaluate import do_evaluate
from predict import do_predict, write_fn, predict_file
from task_label_description import TASK_LABELS_DESC
# from losses import SupConLoss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        help="The task_name to be evaluated")

    parser.add_argument(
        "--use_best_model",
        required=True,
        type=str,
        help="The task_name to be evaluated")

    parser.add_argument(
        "--data_name",
        required=True,
        type=str)
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--negative_num",
        default=1,
        type=int,
        help="Random negative sample number for efl strategy")
    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--save_dir",
        default='./checkpoint',
        type=str,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        "--output_dir",
        default='./predict_output',
        type=str,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--weight_decay", # 0.1
        default=0.1,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--warmup_proportion",
        default=0.06, # 0.06
        type=float,
        help="Linear warmup proption over the training process.")
    parser.add_argument(
        "--init_from_ckpt",
        type=str,
        default=None,
        help="The path of checkpoint to be loaded.")
    parser.add_argument(
        "--seed", type=int, default=1000, help="random seed for initialization")
    parser.add_argument(
        '--device',
        choices=['cpu', 'gpu'],
        default="gpu",
        help="Select which device to train model, defaults to gpu.")
    parser.add_argument(
        '--save_steps',
        type=int,
        default=100000,
        help="Inteval steps to save checkpoint")
    return parser.parse_args()


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def _read_train():
    path = './dataset/{}/train.csv'.format(args.data_name)
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for index, line in enumerate(f):
            line = line.strip()
            id, label, text = line.split('\t')
            yield {'id':id, 'label': label, 'sentence': text}

def _read_dev():
    path = './dataset/{}/dev.csv'.format(args.data_name)
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for index, line in enumerate(f):
            line = line.strip()
            id, label, text = line.split('\t')
            yield {'id':id, 'label': label, 'sentence': text}

def _read_test():
    path = './dataset/{}/test.csv'.format(args.data_name)
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for index, line in enumerate(f):
            id, label, text = line.split('\t')
            yield {'id':id, 'label': label, 'sentence': text}


def do_train():
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)


    train_ds= load_dataset(_read_train,
                           name=args.task_name, #tnews
                           data_files='.dataset/{}/train.csv'.format(args.data_name),
                           splits=None, lazy=False)

    public_test_ds = load_dataset(_read_dev,
                                  name=args.task_name, #tnews
                                  data_files='./dataset/{}/dev.csv'.format(args.data_name),
                                  splits=None, lazy=False)

    test_ds = load_dataset(_read_test,
                           name=args.task_name, #tnews
                           data_files='./dataset/{}/test.csv'.format(args.data_name),
                           splits=None, lazy=False)

    from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
    model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path='skep_ernie_2.0_large_en', num_classes=2)
    tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")#skep_ernie_2.0_large_en, skep_roberta_large_en
    save_dir = 'checkpoint/model_twitter2013'
    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
    state_dict = paddle.load(save_param_path)
    model.set_dict(state_dict)
    # tokenizer.load_pretrained(save_dir)
    # tokenizer = tokenizer.from_pretrained(save_dir)
    print("warmup from:{}".format(args.init_from_ckpt))

    processor = processor_dict[args.task_name](args.negative_num)
    train_ds = processor.get_train_datasets(args.data_name, train_ds,
                                            TASK_LABELS_DESC[args.task_name])

    public_test_ds = processor.get_dev_datasets(args.data_name,
        public_test_ds, TASK_LABELS_DESC[args.task_name])

    test_ds = processor.get_test_datasets(args.data_name, test_ds,
                                          TASK_LABELS_DESC[args.task_name])


    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(dtype="int64"),  # labels
    ): [data for data in fn(samples)]


    predict_batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    ): [data for data in fn(samples)]

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length)

    predict_trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    # train_data_loader = create_dataloader(
    #     train_ds,
    #     mode='train',
    #     batch_size=args.batch_size,
    #     batchify_fn=batchify_fn,
    #     trans_fn=trans_func)
    #
    # public_test_data_loader = create_dataloader(
    #     train_ds,
    #     mode='eval',
    #     batch_size=args.batch_size,
    #     batchify_fn=batchify_fn,
    #     trans_fn=trans_func)

    test_data_loader = create_dataloader(
        train_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=predict_batchify_fn,
        trans_fn=predict_trans_func)


    num_training_steps = len(test_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    # y_pred_labels_test, test_prob, all_prob_test = do_predict(
    #     model,
    #     tokenizer,
    #     test_data_loader,
    #     'test',
    #     args.data_name,
    #     task_label_description=TASK_LABELS_DESC[args.task_name])

    y_pred_labels_train,train_prob, all_prob_train = do_predict(
        model,
        tokenizer,
        test_data_loader,
        'test',
        args.data_name,
        task_label_description=TASK_LABELS_DESC[args.task_name])

    # y_pred_labels_dev, dev_prob, all_prob_dev = do_predict(
    #     model,
    #     tokenizer,
    #     public_test_data_loader,
    #     'dev',
    #     args.data_name,
    #     task_label_description=TASK_LABELS_DESC[args.task_name])



if __name__ == "__main__":
    args = parse_args()
    do_train()
