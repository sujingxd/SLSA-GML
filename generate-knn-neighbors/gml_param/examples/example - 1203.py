import pickle
import time
import warnings

import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
dirname, filename = os.path.split(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(dirname, "..")))

from gml import  GML
from gml_utils import *







import pandas as pd
import pickle
from sklearn import metrics
import os
# from data.gp_split_subgraph import get_unary_cnt_from_fre
import math
import numpy as np
## 提取只有dnn输出的因子+ 距离比值 + knn
import csv


def get_my_all_neighbors(knn, knn_num, is_opp):
    real_new_edges = []
    first = []
    second = []
    third = []
    train_labels = list(pd.read_csv(r'E:\transformers-ori\input_data-3\5%-SST\train.csv', sep='\t')['label'])
    for v0, v1, v2 in knn:
        v1 = eval(v1)
        neighbors = dict()
        for ind, each_v in enumerate(v1):
            # new_edges.append([int(v0), int(v1[ind])])
            neighbors[v1[ind]]= train_labels[v1[ind]]
        if list(neighbors.values()).count(1)> list(neighbors.values()).count(0):
            for ind, each_v in enumerate(v1):
                if train_labels[v1[ind]] == 1:
                    real_new_edges.append([int(v0), int(v1[ind]),1])
                    first.append(int(v0))
                    second.append(int(v1[ind]))
                    third.append(1)
                    break
        else:
            for ind, each_v in enumerate(v1):
                if train_labels[v1[ind]] == 0:
                    real_new_edges.append([int(v0), int(v1[ind]),1])
                    first.append(int(v0))
                    second.append(int(v1[ind]))
                    third.append(1)
                    break
    pdf = pd.DataFrame()
    pdf['id'] = first
    pdf['neighbors'] = second
    pdf.to_csv('new_result.csv', index=None, sep='\t')
    return np.array(real_new_edges)


def get_all_neighbors(knn, knn_num, is_opp):
    new_edges = []
    for v0, v1, v2 in knn:
        v1 = eval(v1)
        v2 = eval(v2)
        for ind, each_v in enumerate(v1):
            new_edges.append([int(v0), int(v1[ind]),v2[ind]])
    return np.array(new_edges)


def get_all_neighbors_train(knn, knn_num, is_opp):
    new_edges = []
    if is_opp:
        value = 0
    else:
        value = 1
    for v0,v1, v_all in knn:
        # v_all = eval(v_all)
        new_edges.append([int(v0), int(v_all), int(value)])
    return np.array(new_edges)

def get_all_neighbors0(knn, knn_num, is_opp):
    new_edges = []
    if is_opp:
        value = 0
    else:
        value = 1
    ids = list(knn['id'])
    neighors = list(knn['neighbors'])
    for v0, v_all in zip(ids, neighors):
        v_all = eval(v_all)
        for each_v in v_all:
            new_edges.append([int(v0), int(each_v), int(value)])
    return np.array(new_edges)

def get_sentiment_tree(arg_text_csv_json):
    import json
    with open(arg_text_csv_json, 'r') as f:
        data = json.load(f)
    expression = []
    pre_labels = []
    sen_value_list = []
    for sen in data['sentences']:
        sen_set = sen["sentimentDistribution"]
        sen_value = sen["sentimentValue"]
        sen_value_list.append(sen_value)
        if int(sen_value) >= 2:
            pre_labels.append(1)
        else:
            pre_labels.append(0)
        expression.append(sen_set)
    pdf = pd.DataFrame()
    pdf['pre_label'] = pre_labels
    return sen_value_list


def after_filter(knn_simi_opp_tmp, all_edges):
    res = []
    edge_dic = dict()
    for edg0,edg1,_ in knn_simi_opp_tmp:
        edge_dic[(edg0, edg1)] = _
    for edg0,edg1,_ in all_edges:
        if (edg0, edg1) in edge_dic:
            continue
        else:
            edge_dic[(edg0, edg1)] = _
    for (k0,k1),v in edge_dic.items():
        res.append([k0,k1,v])
    return np.array(res)


def generate_feature_pickl_only_has_factor(common_name, data_name, data_name_1,TYPE,segment_symbol):
    knn_csv_prefix = r'G:\DNN提取规则\all_result\{}'.format(common_name)
    embedding_csv_prefix = r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data\5%-SST-special\{}'.format(common_name)
    knn_sim0= pd.read_csv(knn_csv_prefix +  r'\SST_knn_9_of_train_bert-base-uncased_train_train.txt', sep='\t').values
    knn_sim1 = pd.read_csv(knn_csv_prefix +  r'\SST_knn_9_of_test_bert-base-uncased.txt', sep='\t').values

    if segment_symbol:
        embedding_train = pd.read_csv(embedding_csv_prefix + r'\train_prediction.csv',sep='\t')['prob'].values
        embedding_test = pd.read_csv(embedding_csv_prefix + r'\test_prediction.csv',sep='\t')['prob'].values
    else:
        embedding_train = pd.read_csv(embedding_csv_prefix + r'\train_prediction.csv',sep=',')['prob'].values
        embedding_test = pd.read_csv(embedding_csv_prefix + r'\test_prediction.csv',sep=',')['prob'].values

    all_data_num = np.concatenate((embedding_train, embedding_test))
    all_data_num  = all_data_num.reshape(all_data_num.shape[0], 1)

    all_edges0 = get_all_neighbors(knn_sim0, 1, is_opp=False)
    all_edges1 = get_all_neighbors(knn_sim1, 1, is_opp=False)

    all_edges = np.append(all_edges0, all_edges1, axis=0)
    var_id_2_feature_ids = dict()
    all_lines = list()
    feature_id_2_weight = dict()
    var_id_2_feature_ids_2 = dict() # 双因子
    var_id_2_feature_ids_2_list = dict()
    feature_id_2_ids = dict()
    for num in range(1):
            line = dict()
            line['feature_id'] = num
            line['feature_type'] = 'unary_feature'
            line['parameterize'] = 1
            line['feature_name'] = 'dnn_{}'.format(num)
            dic_specific = dict()
            for each_num in range(len(all_data_num)):
                temp = all_data_num[each_num][num]
                dic_specific[each_num] = [0.5 if temp >0.5 else -0.5, temp]
                if each_num not in var_id_2_feature_ids:
                    var_id_2_feature_ids[each_num] = list()
                var_id_2_feature_ids[each_num].append(line['feature_id'])
            line['weight'] = dic_specific
            if num in feature_id_2_weight:
                assert  0
            feature_id_2_weight[line['feature_id']] = dic_specific
            all_lines.append(line)
    if data_name_1 == TYPE:
        len_train = len(embedding_train)
        len_test = len(embedding_test)

    first_kind = []
    second_kind = []
    third_kind = []
    fourth_kind = []
    fifth_kind = []
    sixth_kind = []
    seventh_kind = []
    eighth_kind = []

    for (edge0, edge1, value) in all_edges:
        first_kind.append((int(edge0), int(edge1), 1))

    for index, binary in enumerate([first_kind]):
        line = dict()

        line['feature_type'] = 'unary_feature'
        line['parameterize'] = 1
        if index ==0:
            line['feature_name'] = 'sliding_window_factor'
            # line['feature_id'] = all_data_num.shape[1]
            line['feature_id'] = 1
        elif index==1:
            line['feature_name'] = 'second_simi'
            line['feature_id'] = 1
        elif index == 2:
            line['feature_name'] = 'third_simi'
            line['feature_id'] = 2
        else:
            continue
        dic_specific = dict()
        for inx, (edge0, edge1, value) in enumerate(binary):
            if edge0 > len_train + len_test:
                print("error!")
            if edge1 > len_train + len_test:
                print('error!')
            # dic_specific[(edge0, edge1)] = [0.5, 1] if value == 1 else [-0.5, -1]
            if (edge0, edge1) in dic_specific:
                continue
            dic_specific[(edge0, edge1)] = [0.5, value]
            if (edge0, edge1) not in var_id_2_feature_ids_2:
                var_id_2_feature_ids_2[(edge0, edge1)] = list()
            var_id_2_feature_ids_2[(edge0, edge1)].append(line['feature_id'])
            if edge0 not in var_id_2_feature_ids_2_list:
                var_id_2_feature_ids_2_list[edge0] = set()
            if edge1 not in var_id_2_feature_ids_2_list:
                var_id_2_feature_ids_2_list[edge1] = set()
            var_id_2_feature_ids_2_list[edge0].add(line['feature_id'])
            var_id_2_feature_ids_2_list[edge1].add(line['feature_id'])
            if line['feature_id'] not in feature_id_2_ids:
                feature_id_2_ids[line['feature_id']] = list()
            if (edge0, edge1) not in feature_id_2_ids[line['feature_id']] and (edge1, edge0) not in feature_id_2_ids[line['feature_id']]:
                feature_id_2_ids[line['feature_id']].append((edge0, edge1))
        line['weight'] = dic_specific
        feature_id_2_weight[line['feature_id']] = dic_specific
        all_lines.append(line)
    # fw = open('./{}/only_one_'.format(data_name_1) + data_name_1 + "_features.pkl", 'wb')
    # pickle.dump(all_lines, fw)

    return var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids, all_lines

def generate_variable_pickl_only_has_factor(data_name_1, data_name, var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids,embedding_csv_prefix):

    type='special'
    test_info = pd.read_csv(embedding_csv_prefix + r"\test.csv",sep='\t',quoting=csv.QUOTE_NONE)
    train_info = pd.read_csv(embedding_csv_prefix + r"\train.csv",sep='\t',quoting=csv.QUOTE_NONE)

    easy_ids = list(train_info['id'])
    all_ids = list(train_info['id'])+list(test_info['id'])
    all_labels = list(train_info['label'])+list(test_info['label'])
    all_lines = list()
    easy_len = len(easy_ids)
    for elem in range(len(all_ids)):
        line = dict()
        var_id_val = elem
        line['var_id'] = var_id_val
        if var_id_val<easy_len:
            line['is_easy'] = True
        else:
            line['is_easy'] = False
        line['label'] = all_labels[elem] if line['is_easy'] == True else -1
        line['is_evidence'] = line['is_easy']
        line['true_label'] = all_labels[elem]
        line['prior'] = 0
        feature_dic = dict()
        if line['var_id'] in var_id_2_feature_ids:
            feature_id_list = var_id_2_feature_ids[line['var_id']]
            for feature_id in feature_id_list:
                if feature_id in feature_id_2_weight:
                    temp_list = list(feature_id_2_weight[feature_id][line['var_id']])
                final_list = [0.5 if temp_list[1]>0.5 else -0.5, temp_list[1]]
                feature_dic[feature_id] = final_list

        if line['var_id'] in var_id_2_feature_ids_2_list:
            for featureid in var_id_2_feature_ids_2_list[line['var_id']]:
                for (vid0,vid1),value in feature_id_2_weight[featureid].items():
                    if vid0 == line['var_id'] or vid1 == line['var_id']:
                        if featureid not in feature_dic:
                            feature_dic[featureid] = list()
                        if (vid0,vid1) not in feature_dic[featureid] and (vid1,vid0) not in feature_dic[featureid]:
                            feature_dic[featureid].append((vid0,vid1))
        line['feature_set'] = feature_dic
        all_lines.append(line)

    # fw = open('./{}/only_one_'.format(data_name_1) + data_name_1 + "_variables.pkl", 'wb')
    # pickle.dump(all_lines, fw)
    return all_lines
# if __name__ == "__main__":


def perform_dnn(common_name, segment_symbol):
    knn_csv_prefix = r'G:\DNN提取规则\all_result\{}'.format(common_name)
    embedding_csv_prefix = r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data\5%-SST-special\{}\\'.format(common_name)
    data_name = data_name_1 = knn_csv_prefix.split("\\")[-1]
    TYPE = data_name
    # os.mkdir("../data/"+'')
    var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids,feature_pkl = generate_feature_pickl_only_has_factor(common_name,data_name,data_name_1,TYPE,segment_symbol)
    var_pkl = generate_variable_pickl_only_has_factor(data_name_1, data_name, var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids,embedding_csv_prefix)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    warnings.filterwarnings('ignore')  # 过滤掉warning输出
    begin_time = time.time()
    # 1.准备数据
    dataname = data_name
    variables = var_pkl
    features = feature_pkl
    opt=None
    graph = GML.initial("example.config", opt, variables, features)
    #3. 因子图推理
    graph.inference()
    #可选： 4.保存推理后的因子图模型
    # graph.save_model()
    #4. 输出推理用时
    end_time = time.time()
    print('Running time: %s Seconds' % (end_time - begin_time))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
