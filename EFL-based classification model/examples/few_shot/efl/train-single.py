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
# def parse_args(save_dir, output_dir, seed, data,use_best_model):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        required=True,
        type=str,
        default='eprstmt',
        help="The task_name to be evaluated")

    parser.add_argument(
        "--use_best_model",
        required=True,
        # default=use_best_model,
        type=str,
        help="The task_name to be evaluated")

    parser.add_argument(
        "--data_name",
        required=True,
        # default=data,
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
        # default=2e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--save_dir",
        default='./twitter2013/',
        type=str,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        "--output_dir",
        default='./twitter2013/',
        type=str,
        help="The output directory where the model checkpoints will be written.")
    parser.add_argument(
        "--max_seq_length",
        default=128,
        # default=100,
        type=int,
        help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--weight_decay", # 0.1
        # default=0.2,
        default=0.2,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--epochs",
        default=5,
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
        "--seed", type=int, help="random seed for initialization")
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

    # train_ds, public_test_ds, test_ds = load_dataset(
    #     "fewclue",
    #     name=args.task_name,
    #     splits=("train_0", "test_public", "test"))


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

    # model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
    #     'ernie-1.0', num_classes=2)
    # tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')
    from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
    model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path='skep_ernie_2.0_large_en', num_classes=2)
    tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")#skep_ernie_2.0_large_en, skep_roberta_large_en

    processor = processor_dict[args.task_name](args.negative_num)
    train_ds = processor.get_train_datasets(args.data_name, train_ds,
                                            TASK_LABELS_DESC[args.task_name])

    public_test_ds = processor.get_dev_datasets(args.data_name,
        public_test_ds, TASK_LABELS_DESC[args.task_name])

    test_ds = processor.get_test_datasets(args.data_name, test_ds,
                                          TASK_LABELS_DESC[args.task_name])

    # [src_ids, token_type_ids, labels]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # src_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(dtype="int64"),  # labels
    ): [data for data in fn(samples)]

    # [src_ids, token_type_ids]
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

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    public_test_data_loader = create_dataloader(
        public_test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    test_data_loader = create_dataloader(
        test_ds,
        mode='eval',
        batch_size=args.batch_size,
        batchify_fn=predict_batchify_fn,
        trans_fn=predict_trans_func)

    # if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
    #     state_dict = paddle.load(args.init_from_ckpt)
    #     model.set_dict(state_dict)
    #     print("warmup from:{}".format(args.init_from_ckpt))

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    # temp = 0.01
    # criterion = paddle.nn.loss.SupConLoss()
    global_step = 0
    tic_train = time.time()
    best_dev_accurcy = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, batch in enumerate(train_data_loader, start=1):

            src_ids, token_type_ids, labels = batch

            prediction_scores,  content, pooled_output_skepmodel, pooled_output = model(
                input_ids=src_ids, token_type_ids=token_type_ids)

            loss = criterion(prediction_scores, labels)

            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, step, loss,
                       10 / (time.time() - tic_train)))
                tic_train = time.time()

            # if global_step % args.save_steps == 0 and rank == 0:
            #     save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
            #     save_param_path = os.path.join(save_dir, 'model_state.pdparams')
            #     paddle.save(model.state_dict(), save_param_path)
            #     tokenizer.save_pretrained(save_dir)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

        test_public_accuracy, total_num = do_evaluate(
            model,
            tokenizer,
            public_test_data_loader,
            task_label_description=TASK_LABELS_DESC[args.task_name])

        if args.use_best_model:
            if test_public_accuracy > best_dev_accurcy:
                best_dev_accurcy = test_public_accuracy
                if not os.path.exists('fintuning_model'):
                    os.makedirs('fintuning_model')
                save_path = 'fintuning_model/{}/{}_model_state.pdparams'.format(args.data_name,args.data_name)
                paddle.save(model.state_dict(), save_path)
                tokenizer.save_pretrained('fintuning_model/{}/'.format(args.data_name))
                import copy
                best_model = copy.deepcopy(model)
                state_dict = paddle.load(save_path)
                best_model.set_dict(state_dict)

        print("epoch:{}, dev_accuracy:{:.3f}, total_num:{}".format(
            epoch, test_public_accuracy, total_num))

        y_pred_labels, _, _ = do_predict(
            model,
            tokenizer,
            test_data_loader,
            'test',
            'no',
            task_label_description=TASK_LABELS_DESC[args.task_name])

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_file = os.path.join(args.output_dir,
                                   str(epoch) + predict_file[args.task_name])

        write_fn[args.task_name](args.task_name, output_file, y_pred_labels)

        # if rank == 0:
        #     save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     save_param_path = os.path.join(save_dir, 'model_state.pdparams')
        #     paddle.save(model.state_dict(), save_param_path)
        #     tokenizer.save_pretrained(save_dir)
        import pandas as pd
        test_label = pd.read_csv('./dataset/{}/test.csv'.format(args.data_name), sep='\t')['label'].values
        train_label = pd.read_csv('./dataset/{}/train.csv'.format(args.data_name), sep='\t')['label'].values
        dev_label = pd.read_csv('./dataset/{}/dev.csv'.format(args.data_name), sep='\t')['label'].values
        # y_pred_labels = [1 if i == 'Positive' else 0 for i in y_pred_labels]
        y_pred_labels_new = np.array(y_pred_labels)
        print("test acc : {}".format((test_label == y_pred_labels_new).mean()))
        write_fn[args.task_name](args.task_name, output_file, y_pred_labels)

    if args.use_best_model:
        model = best_model

    save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
    paddle.save(model.state_dict(), save_param_path)
    tokenizer.save_pretrained(save_dir)
    y_pred_labels_test, test_prob, all_prob_test = do_predict(
        model,
        tokenizer,
        test_data_loader,
        'test',
        args.data_name,
        task_label_description=TASK_LABELS_DESC[args.task_name])

    y_pred_labels_train,train_prob, all_prob_train = do_predict(
        model,
        tokenizer,
        train_data_loader,
        'train',
        args.data_name,
        task_label_description=TASK_LABELS_DESC[args.task_name])

    y_pred_labels_dev, dev_prob, all_prob_dev = do_predict(
        model,
        tokenizer,
        public_test_data_loader,
        'dev',
        args.data_name,
        task_label_description=TASK_LABELS_DESC[args.task_name])

    # y_pred_labels_test = [1 if i == 'Positive' else 0 for i in y_pred_labels_test]
    # y_pred_labels_train = [1 if i == 'Positive' else 0 for i in y_pred_labels_train]
    # y_pred_labels_dev = [1 if i == 'Positive' else 0 for i in y_pred_labels_dev]
    print("data_name is : {}\n".format(args.data_name))
    print("seed is : {}\n".format(args.seed))
    print("best test acc : {}".format((test_label == y_pred_labels_test).mean()))
    print("best train acc : {}".format((train_label == y_pred_labels_train).mean()))
    print("best dev acc : {}".format((dev_label == y_pred_labels_dev).mean()))

    # file_write = open(args.data_name+"_acc.txt",'w')
    file_write = open("all_acc.txt", 'a')
    file_write.write("data_name is : {}\n".format(args.data_name))
    file_write.write("seed is : {}\n".format(args.seed))
    file_write.write("best train acc : {}".format((train_label == y_pred_labels_train).mean()))
    file_write.write("best dev acc : {}".format((dev_label == y_pred_labels_dev).mean()))
    file_write.write("best test acc : {}".format((test_label == y_pred_labels_test).mean()))
    from sklearn import metrics
    precision = metrics.precision_score(test_label, y_pred_labels_test)
    recall = metrics.recall_score(test_label, y_pred_labels_test)
    f1_score = metrics.f1_score(test_label, y_pred_labels_test)
    file_write.write("best_test_precision is: {}".format(precision))
    file_write.write("best_test_recall is: {}".format(recall))
    file_write.write("best_f1 score is: {}".format(f1_score)+"\n")
    print("best_test_precision is: {}".format(precision))
    print("best_test_recall is: {}".format(recall))
    print("best_f1 score is: {}".format(f1_score)+"\n")
    file_prob_train_name = args.data_name+"train_ouput_prob.txt"
    file_prob_dev_name = args.data_name+"dev_ouput_prob.txt"
    file_prob_test_name = args.data_name+"test_ouput_prob.txt"
    pd_train = pd.DataFrame()
    pd_train['id'] = list(range(len(train_prob)))
    pd_train['prob'] = train_prob
    pd_train['label'] = train_label
    pd_train.to_csv(file_prob_train_name, index=None)
    pd_dev = pd.DataFrame()
    pd_dev['id'] = list(range(len(dev_prob)))
    pd_dev['prob'] = dev_prob
    pd_dev['label'] = dev_label
    pd_dev.to_csv(file_prob_dev_name, index=None)
    pd_test = pd.DataFrame()
    pd_test['id'] = list(range(len(test_prob)))
    pd_test['prob'] = test_prob
    pd_test['label'] = test_label
    pd_test.to_csv(file_prob_test_name, index=None)

    all_file_prob_train_name = args.data_name+"train_ouput_prob_all.txt"
    all_file_prob_dev_name = args.data_name+"dev_ouput_prob_all.txt"
    all_file_prob_test_name = args.data_name+"test_ouput_prob_all.txt"
    pd_train_all = pd.DataFrame()
    pd_train_all['id'] = list(range(len(all_prob_train)))
    pd_train_all['prob'] = all_prob_train
    pd_train_all.to_csv(all_file_prob_train_name, index=None)
    pd_dev_all = pd.DataFrame()
    pd_dev_all['id'] = list(range(len(all_prob_dev)))
    pd_dev_all['prob'] = all_prob_dev
    pd_dev_all.to_csv(all_file_prob_dev_name, index=None)
    pd_test_all = pd.DataFrame()
    pd_test_all['id'] = list(range(len(all_prob_test)))
    pd_test_all['prob'] = all_prob_test
    pd_test_all.to_csv(all_file_prob_test_name, index=None)



if __name__ == "__main__":
    # data_name = ['SST','CR','MR','twitter2013']
    # use_best_model = ['True','False']
    # seeds_all = [42, 20]
    #
    # for data in data_name:
    #     for seed in seeds_all:
    #         output_dir = data +"_"+str(seed)+ "/"
    #         args = parse_args(output_dir, output_dir, seed, data,use_best_model)
    #         do_train()
    args= parse_args()
    do_train()