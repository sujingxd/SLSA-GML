#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle

from model import CrossEntropyWithKL, VAESeq2SeqModel, Perplexity, NegativeLogLoss, TrainCallback
from args import parse_args
from data import create_data_loader, my_create_data_loader
import numpy as np

def  write_this_sentence(file_out, text):
    for i in list(range(2262)):
        file_out.write(text)
    file_out.close()


def calc_prob_belong_to_this_distribution(e_0, var_0, e_point_all):
    """
     Compute the KL divergence between Gaussian distribution
     """
    all_prob_value = []
    for each_point in e_point_all:
        prob_value = 1/(math.sqrt(2*math.pi)*var_0)* np.exp(-pow((e_point-e_0),2)/(2*pow(var_0,2)))
        all_prob_value.append(prob_value)

    return all_prob_value



def calc_kl_dvg(e_0, var_0, e_1, var_1):
    """
     Compute the KL divergence between Gaussian distribution
     """
    kl_cost = np.log(np.absolute(var_1/var_0))+(np.power(var_0,2)+np.power((e_0-e_1),2))/2*np.power(var_1,2)-1/2
    kl_cost = np.mean(kl_cost, 0)

    return np.sum(kl_cost)

def save_matrix(arr_result, file_name):
    f_out = open(file_name, 'w')
    np.set_printoptions(threshold=np.inf)
    np.savetxt(f_out, arr_result, fmt='%.17f')
    f_out.close()

def train(args):
    print(args)
    device = paddle.set_device(args.device)
    train_loader, dev_loader, test_loader, vocab, bos_id, pad_id, train_data_len = create_data_loader(
        args)

    net = VAESeq2SeqModel(
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        vocab_size=len(vocab) + 2,
        num_layers=args.num_layers,
        init_scale=args.init_scale,
        enc_dropout=args.enc_dropout,
        dec_dropout=args.dec_dropout)

    gloabl_norm_clip = paddle.nn.ClipGradByGlobalNorm(args.max_grad_norm)

    anneal_r = 1.0 / (args.warm_up * train_data_len / args .batch_size)
    cross_entropy = CrossEntropyWithKL(
        base_kl_weight=args.kl_start, anneal_r=anneal_r)
    model = paddle.Model(net)

    optimizer = paddle.optimizer.Adam(
        args.learning_rate,
        parameters=model.parameters(),
        grad_clip=gloabl_norm_clip)

    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    ppl_metric = Perplexity(loss=cross_entropy)
    nll_metric = NegativeLogLoss(loss=cross_entropy)

    model.prepare(
        optimizer=optimizer,
        loss=cross_entropy,
        metrics=[ppl_metric, nll_metric])

    model.fit(train_data=train_loader,
              eval_data=dev_loader,
              epochs=args.max_epoch,
              save_dir=args.model_path,
              shuffle=False,
              callbacks=[TrainCallback(ppl_metric, nll_metric, args.log_freq)],
              log_freq=args.log_freq)

    # Evaluation
    print('Start to evaluate on test dataset...')
    model.evaluate(test_loader, log_freq=len(test_loader))
    kl_loss, dec_output, trg_mask, z_mean, z_log_var,_ = model.predict(train_loader)
    # output_mean = z_mean
    # output_var = z_log_var
    output_mean = np.concatenate(z_mean,axis=0)
    output_var = np.concatenate(z_log_var,axis=0)
    save_matrix(output_mean, "mean_output.csv")
    save_matrix(output_var, "var_output.csv")
    temp_mean = output_mean
    temp_var = output_var

    _, _, _, _, _,embedding_out = model.predict(test_loader)

    return temp_mean, temp_var,embedding_out
if __name__ == '__main__':
    import  numpy as np
    args = parse_args()
    _,_,embedding_out = train(args)
    e_0 = np.loadtxt(r'/home/ssd1/sj/paddle-yuanshi/PaddleNLP-develop/examples/text_generation/vae-seq2seq/mean_output.csv')
    var_0 = np.loadtxt(r'/home/ssd1/sj/paddle-yuanshi/PaddleNLP-develop/examples/text_generation/vae-seq2seq/var_output.csv')
    # file_input = open('/home/ssd1/sj/paddle-yuanshi/PaddleNLP-develop/examples/text_generation/vae-seq2seq/test.txt', 'r')
    # test_txt = file_input.readlines()
    # total_kl_ouput_content = []
    # new_output_ids = []
    # count = 2262
    # for i in test_txt:
        # file_out = open('/home/ssd1/sj/paddle-yuanshi/PaddleNLP-develop/examples/text_generation/vae-seq2seq/test.txt', 'w')
        # write_this_sentence(file_out, i)
    # temp_mean, temp_var, embedding_out = train(args)
    # kl_temp = calc_kl_dvg(embedding_out,  e_0, var_0)
    prob_value = calc_prob_belong_to_this_distribution(e_0, var_0,embedding_out)
    # new_output_ids.append(count)
    # total_kl_ouput_content.append(prob_value)
    # count += 1
    print('******* this is '+str(count-2262)+"************")
    import pandas as pd
    outfile = pd.DataFrame()
    outfile['id'] = list(range(2262,2262+754))
    outfile['KL_value'] = prob_value
    outfile.to_csv('total_kl_output.csv', sep='\t',index=None)
