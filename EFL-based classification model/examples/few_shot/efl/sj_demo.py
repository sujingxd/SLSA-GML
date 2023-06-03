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
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
import pandas as pd
import pickle

if __name__ == "__main__":
    save_dir = 'checkpoint/model_twitter2013'
    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
    model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path='skep_ernie_2.0_large_en', num_classes=2)
    tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")#skep_ernie_2.0_large_en, skep_roberta_large_en
    state_dict = paddle.load(save_param_path)
    model.set_dict(state_dict)
    tokenizer.load_pretrained(save_dir)
    all_texts = pd.read_csv('dataset/CR/train.csv', sep='\t')['text']
    all_content = []
    all_pooled_output_skepmodel = []
    for each_text in all_texts:
        inputs = tokenizer(each_text)
        labels = paddle.to_tensor([1]).unsqueeze(0)  # Batch size 1
        prediction_probs, content, pooled_output_skepmodel,sentence_out = model(**inputs, labels=labels)
        all_content.append(content)
        all_pooled_output_skepmodel.append(pooled_output_skepmodel)
    content_file_out = open('CR_content.pkl', 'wb')
    skep_model_file_out = open('CR_pool.pkl','wb')
    pickle.dump(all_content,content_file_out)
    pickle.dump(all_pooled_output_skepmodel,skep_model_file_out)




