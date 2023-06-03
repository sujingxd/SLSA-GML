import pandas as pd
import pickle
from sklearn import metrics
import os
from data.gp_split_subgraph import get_unary_cnt_from_fre
import math
import numpy as np
## 提取只有dnn输出的因子+ 距离比值 + knn
import csv


def get_my_all_neighbors(knn, knn_num, is_opp):
    real_new_edges = []
    first = []
    second = []
    third = []
    train_labels = list(pd.read_csv(r'E:\transformers-ori\input_data-3\5%-MR\train.csv', sep='\t')['label'])
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


def generate_feature_pickl_only_has_factor(data_name, data_name_1,TYPE,segment_symbol):
    # prob_train_info = pd.read_csv(r"F:\codes-KBS\MR_train_prediction.csv".format(data_name))['prob'].values
    # prob_test_info = pd.read_csv(r"F:\codes-KBS\MR_test_prediction.csv".format(data_name))['prob'].values
    # new_interval_factor = pd.read_csv('G:\DNN提取规则\MR_new_add_factor.csv').values
    # new_add_feature_2_id_2_list = pd.read_csv('G:\DNN提取规则\MR_feature_id.csv').values
    # prob_train_info_0 = np.loadtxt(r"G:\DNN提取规则\MR-all_distance_train_all_distance_10.csv")
    # prob_test_info_0 = np.loadtxt(r"G:\DNN提取规则\MR-all_distance_test_all_distance_10.csv")
    # prob_train_info = pd.read_csv(r"G:\transformers-4.3.3\input_data-3-1-10\MR-bert-new\train_prediction.csv".format(data_name))['prob'].values
    # prob_test_info = pd.read_csv(r"G:\transformers-4.3.3\input_data-3-1-10\MR-bert-new\test_prediction.csv".format(data_name))['prob'].values

    # prob_train_info_0 = np.loadtxt(r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\MR-all\bert-base-cased\MR_distance_train_sentence_out.csv")
    # prob_test_info_0 = np.loadtxt(r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\MR-all\bert-base-cased\MR_distance_test_sentence_out.csv")


    # prob_train_info = np.array(get_sentiment_tree(r'E:\gml-old-平台-0\0712-data_input\train_text.txt.json'))
    # prob_test_info = np.array(get_sentiment_tree(r'E:\gml-old-平台-0\0712-data_input\test_text.txt.json'))
    # prob_train_info = np.loadtxt(r"E:\gml-old-平台-0\data\MR\distance_MR_train_ss&sp.csv")
    # prob_test_info = np.loadtxt('E:\gml-old-平台-0\data\MR\distance_MR_ss&sp.csv')

    # prob_train_info = pd.read_csv(r"G:\transformers-4.3.3\new_datasets\MR-bert\bert-base-uncased\train_prediction.csv".format(data_name)).values[:, 2]
    # prob_test_info = pd.read_csv(r"G:\transformers-4.3.3\new_datasets\MR-bert\bert-base-uncased\test_prediction.csv".format(data_name)).values[:, 2]

    # prob_train_info_0 = pd.read_csv(r'G:\transformers-4.3.3\input_data-3-1-10\MR-bert\bert-base-uncased\train_prediction.csv').values[:, 2]
    # prob_test_info_0 =  pd.read_csv(r'G:\transformers-4.3.3\input_data-3-1-10\MR-bert\bert-base-uncased\test_prediction.csv').values[:, 2]
    # prob_train_info_0 = prob_train_info_0.reshape(prob_train_info_0.shape[0],1)
    # prob_test_info_0 = prob_test_info_0.reshape(prob_test_info_0.shape[0], 1)

    # prob_train_info = np.loadtxt(r"G:\DNN提取规则\MR_distance_train_all_distance_20.csv".format(data_name))
    # prob_test_info = np.loadtxt(r"G:\DNN提取规则\MR_distance_test_all_distance_20.csv".format(data_name))

    # prob_train_info = np.loadtxt(r"L:\generating-reviews-discovering-sentiment-master\MR_train_predict_y.csv".format(data_name))
    # prob_test_info = np.loadtxt(r"L:\generating-reviews-discovering-sentiment-master\MR_test_predict_y.csv".format(data_name))

    # info0 = np.loadtxt(r"E:\stc_clustering-master\data\MR\text_matrix_structure.txt".format(data_name))
    # train_info0 = info0[:2262]
    # test_info0 = info0[3016:]
    # info1 = np.loadtxt(r"E:\stc_clustering-master\data\MR\text_matrix_embedding.txt".format(data_name))
    # train_info01 = info1[:2262]
    # test_info1 = info1[3016:]
    # prob_train_info = np.concatenate((train_info0,train_info01),axis=1)
    # prob_test_info = np.concatenate((test_info0, test_info1),axis=1)


    # info = pd.read_csv("E:\stc_clustering-master\data\MR\stc.csv").values
    # prob_train_info = info[:8534]
    # prob_test_info = info[9612:]
    # print("**")
    # prob_train_info = np.loadtxt(r"G:\DNN提取规则\MR_distance_train_all_distance_7.csv".format(data_name))
    # prob_test_info = np.loadtxt(r"G:\DNN提取规则\MR_distance_test_all_distance_7.csv".format(data_name))

    # prob_train_info = pd.read_csv(r"F:\codes-KBS\MR_train_prediction.csv".format(data_name)).values[:, 1]
    # prob_test_info = pd.read_csv( r"F:\codes-KBS\MR_test_prediction.csv".format(data_name)).values[:, 1]
    # prob_train_info = prob_train_info.reshape(prob_train_info.shape[0],1)
    # prob_test_info = prob_test_info.reshape(prob_test_info.shape[0], 1)
    # attention_train_info = np.loadtxt(r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\MR-all\xlnet-base-cased\MR_distance_train_all_attention.csv".format(data_name))
    # max_pooling_all_test_info = np.loadtxt( r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\MR-all\xlnet-base-cased\MR_distance_test_max_pooling_all.csv".format(data_name))
    # pooled_output_all_train_info = np.loadtxt(r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\MR-all\xlnet-base-cased\MR_distance_train_pooled_output_all.csv".format(data_name))
    # pooled_output_all_test_info = np.loadtxt( r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\MR-all\xlnet-base-cased\MR_distance_test_pooled_output_all.csv".format(data_name))

    # prob_train_info = pd.read_csv(r'G:\Subpattern_rule_3_21\MR-4个成分\new_four_elements_train_srl_info-1_words_table.csv'.format(data_name)).values
    # prob_test_info = pd.read_csv(r'G:\Subpattern_rule_3_21\MR-4个成分\new_four_elements_test_srl_info-1_words_table.csv'.format(data_name)).values

    # knn_sim0 = pd.read_csv(r'G:\DNN提取规则\all_result\k-5-test-近邻\{}_knn_5_of_test_bert-base-uncased.txt'.format((data_name)), sep='\t').values
    # knn_sim1 = pd.read_csv(r'G:\DNN提取规则\all_result\k-5-train-近邻\{}_knn_1_of_test_bert-base-uncased.txt'.format((data_name)), sep='\t').values


    # knn_sim0= pd.read_csv(r'G:\DNN提取规则\all_result\5%-MR\MR_knn_20_of_train_bert-base-uncased_train_train.txt'.format((data_name)), sep='\t').values
    # knn_sim1 = pd.read_csv(r'G:\DNN提取规则\all_result\5%-MR\MR_knn_20_of_test_bert-base-uncased.txt'.format((data_name)), sep='\t').values

    # type='new-add-test
    # type='special'
    knn_sim0= pd.read_csv(knn_csv_prefix +  r'\MR_knn_10_of_train_bert-base-uncased_train_train.txt', sep='\t').values
    knn_sim1 = pd.read_csv(knn_csv_prefix +  r'\MR_knn_10_of_test_bert-base-uncased.txt', sep='\t').values
    # knn_sim2 = pd.read_csv(r'G:\DNN提取规则\all_result\语义角色标注因子\{}_knn_5_of_test_bert-base-uncased.txt'.format((data_name)),sep='\t').values

    # embedding_train = pd.read_csv(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\1%-MR\roberta-basetrain_prediction.csv')['prob'][:727]
    # embedding_test = pd.read_csv(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\1%-MR\roberta-basetest_prediction.csv')['prob'].values

    # embedding_train = pd.read_csv(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\1%-MR-right-now\train_prediction.csv')['prob'][:347]
    # embedding_test = pd.read_csv(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\1%-MR-right-now\test_prediction.csv')['prob'].values
    if segment_symbol:
        embedding_train = pd.read_csv(embedding_csv_prefix + r'train_prediction.csv',sep='\t')['prob'].values
        embedding_test = pd.read_csv(embedding_csv_prefix + r'test_prediction.csv',sep='\t')['prob'].values
    else:
        embedding_train = pd.read_csv(embedding_csv_prefix + r'train_prediction.csv')['prob'].values
        embedding_test = pd.read_csv(embedding_csv_prefix + r'test_prediction.csv')['prob'].values
    # embedding_dev = pd.read_csv(embedding_csv_prefix + r'dev_prediction.csv',sep='\t')['prob'].values

    all_data_num = np.concatenate((embedding_train, embedding_test))
    all_data_num  = all_data_num.reshape(all_data_num.shape[0], 1)
    # knn_sim0 = pd.read_csv(r'G:\DNN提取规则\all_result\5%-MR-激活因子\MR_knn_2_of_train_bert-base-uncased_train_train.txt'.format((data_name)), sep='\t').values
    # knn_sim1 = pd.read_csv(r'G:\DNN提取规则\all_result\5%-MR-激活因子\MR_knn_8_of_test_bert-base-uncased.txt'.format((data_name)), sep='\t').values


    # embedding_test = pd.read_csv(r"E:\transformers-ori\input_data-3-1-10\{}\df_test_factors.csv".format(data_name)).values[:,4:]
    # embedding_train = pd.read_csv(r"E:\transformers-ori\input_data-3-1-10\{}\df_train_factors.csv".format(data_name)).values[:,4:]

    # feature_num = embedding_train.shape[1] # 一共768维度特征个数
    # all_data_num = np.concatenate((embedding_train, embedding_test))

    # knn_3 = np.loadtxt('G:\DNN提取规则\{}_lnn_all_neighors_8_of_train_xlnet-base-cased.txt'.format((data_name)))
    # knn_1_test = np.loadtxt('G:\DNN提取规则\{}_lnn_all_neighors_5_of_test_xlnet-base-cased.txt'.format((data_name)))
    # knn_3_test = np.loadtxt('G:\DNN提取规则\{}_lnn_all_neighors_8_of_test_xlnet-base-cased.txt'.format((data_name)))

    # knn_simi_opp0 = pd.read_csv(r'E:\gml-old-平台-0\data\MR\binary_relations-0618.csv'.format((data_name))).values
    # knn_1 = np.loadtxt(r'G:\DNN提取规则\MR_lnn_all_neighors_1_of_test_bert-base-uncased.txt'.format(data_name), dtype='int')
    # knn_2 = np.loadtxt(r'G:\DNN提取规则\all_result\MR_knn_3_of_test_bert-base-uncased-oppo.txt', dtype='int')
    # knn_oppo = np.loadtxt(r'G:\DNN提取规则\all_result\all_oppo_relation_MR_knn_1_of_test_MR.txt', dtype='int')
    # # knn_oppo = np.loadtxt(r'G:\DNN提取规则\all_result\MR_knn_3_of_test_bert-base-uncased-newoppo.txt', dtype='int')
    #
    # knn_oppo = pd.read_csv(r'G:\DNN提取规则\all_result\all_oppo_relation_{}_knn_1_of_test_{}.txt'.format(data_name, data_name),sep='\t').values
    # knn_sim0 = pd.read_csv(r'G:\knn-sj\new-jiayi\{}_lnn_all_neighors_1_of_test_O.txt'.format(data_name), sep='\t').values
    # knn_sim1 = pd.read_csv(r'G:\knn-sj\new-jiayi\{}_lnn_all_neighors_1_of_test_argm_of_v.txt'.format(data_name), sep='\t').values
    # knn_sim2 = pd.read_csv(r'G:\knn-sj\new-jiayi\{}_lnn_all_neighors_1_of_test_C_of_v.txt'.format(data_name),sep='\t').values
    # knn_sim3 = pd.read_csv(r'G:\knn-sj\new-jiayi\{}_lnn_all_neighors_1_of_test_left_arg_of_v.txt'.format(data_name), sep='\t').values
    # knn_sim4 = pd.read_csv(r'G:\knn-sj\new-jiayi\{}_lnn_all_neighors_1_of_test_R_of_v.txt'.format(data_name), sep='\t').values
    # knn_sim5 = pd.read_csv(r'G:\knn-sj\new-jiayi\{}_lnn_all_neighors_1_of_test_right_arg_of_v.txt'.format(data_name),sep='\t').values
    # knn_sim6 = pd.read_csv(r'G:\knn-sj\new-jiayi\{}_lnn_all_neighors_1_of_test_v.txt'.format(data_name),sep='\t').values
    # knn_oppo = pd.read_csv(r'L:\gml-ER\sentilare_data\MR\bert-base-uncased\MR_oppo.csv',sep='\t').values
    # knn_sim = pd.read_csv(r'L:\gml-ER\sentilare_data\MR\bert-base-uncased\MR_simi.csv', sep='\t').values
    # all_edges_opp = get_all_neighbors(knn_oppo, 1, is_opp=True)
    all_edges0 = get_all_neighbors(knn_sim0, 1, is_opp=False)
    all_edges1 = get_all_neighbors(knn_sim1, 1, is_opp=False)
    # all_edges2 = get_all_neighbors(knn_sim2, 1, is_opp=False)
    # all_edges3 = get_all_neighbors(knn_sim3, 1, is_opp=False)
    # all_edges4 = get_all_neighbors(knn_sim4, 1, is_opp=False)
    # all_edges5 = get_all_neighbors(knn_sim5, 1, is_opp=False)
    # all_edges6 = get_all_neighbors(knn_sim6, 1, is_opp=False)
    # all_edges7 = get_all_neighbors0(knn_1, 1, is_opp=False)
    all_edges = np.append(all_edges0, all_edges1, axis=0)
    # all_edges = all_edges0
    # all_edges = all_edges1
    # all_edges = all_edges1
    # np.savetxt("all_edges.txt",all_edges_smi,fmt="%s", delimiter="\t")
    # all_edges = all_edges_smi
    # my_binary_relation = pd.read_csv(r'G:\jiayi-sujin\sujin\datasets\MR\MR_test_results_only_3.csv',sep='\t').values
    # my_binary_relation = pd.read_csv(r'G:\jiayi-sujin\sujin\datasets\{}\{}_test_results_only_3.csv'.format(data_name, data_name),sep='\t').values
    # my_binary_relation = pd.read_csv(r'G:\jiayi-sujin\sujin\datasets\{}\100\{}_test_results_only_3.csv'.format(data_name, data_name),sep='\t').values
    # my_binary_relation0 = pd.read_csv(r'G:\jiayi-sujin\sujin\datasets\MR\MR_train_train.csv'.format(data_name, data_name),sep='\t').values
    # my_binary_relation  = np.append(my_binary_relation,my_binary_relation0,axis=0)
    # my_binary_relation = pd.read_csv(r'G:\sentence-pair\SentiLARE-master-modify\data\MR\MR_test_results_2_only_3.csv', sep='\t').values
    # my_binary_relation = pd.read_csv(r'E:\transformers-4.3.3-zcl\input_data-3-1-10\MR\res_only_3.csv', sep='\t').values
    # knn_31 = np.loadtxt('G:\DNN提取规则\MR-bert_lnn_all_neighors_3_of_test_bert-base-uncased1.txt', dtype='int')
    # knn_32 = np.loadtxt('G:\DNN提取规则\MR-bert_lnn_all_neighors_3_of_test_bert-base-uncased2.txt', dtype='int')
    # knn_simi_opp0 = pd.read_csv(r'E:\gml-old-平台-0\data\MR\binary_relations_MR.csv'.format(data_name, data_name),sep=',').values
    # my_binary = pd.read_csv(r'E:\gml-old-平台-0\data\MR\MR_test_results0.csv').values
    # knn_simi_opp0 = pd.read_csv()
    # attention_train_info = attention_train_info.reshape(attention_train_info.shape[0], 1)
    # attention_test_info = attention_test_info.reshape(attention_test_info.shape[0], 1)
    # max_pooling_all_train_info = max_pooling_all_train_info.reshape(max_pooling_all_train_info.shape[0], 1)
    # max_pooling_all_test_info = max_pooling_all_test_info.reshape(max_pooling_all_test_info.shape[0], 1)
    # pooled_output_all_train_info = pooled_output_all_train_info.reshape(pooled_output_all_train_info.shape[0], 1)
    # pooled_output_all_test_info = pooled_output_all_test_info.reshape(pooled_output_all_test_info.shape[0], 1)
    # learning_interpretable_metric =pd.read_csv(r'G:\DNN提取规则\MR_lnn_all_neighors_1_of_test_learning_interpretable_metric_distance.csv'.format(data_name, data_name),sep=',').values
    #

    # # prob_test_info = np.append(train_xlnet_res,pooled_output_all_train_info,axis=1)

    # test_roberta_res = np.append(test_roberta_res,pooled_output_all_test_info,axis=1)

    # prob_train_info = np.loadtxt(r'G:\SentiLARE-master\input_data-3-1-10\MR\train_rule_single.txt'.format(data_name), dtype='int')
    # prob_test_info = np.loadtxt(r'G:\SentiLARE-master\input_data-3-1-10\MR\test_rule_single.txt'.format(data_name), dtype='int')
    # # all_data_test = np.loadtxt('E:\gml-old-平台-0\data\MR\distance_MR_ss&sp.csv')
    # prob_train_info = np.append(prob_train_info, prob_train_info, axis=1)
    # prob_test_info = np.append(prob_test_info, prob_test_info, axis=1)
    # prob_train_info = np.loadtxt(r'F:\codes-KBS\MR_train_first_all.csv'.format(data_name), dtype='int')
    # prob_test_info = np.loadtxt(r'F:\codes-KBS\MR_test_first_all.csv'.format(data_name), dtype='int')
    # prob_train_info_0 = np.loadtxt(r'F:\codes-KBS\MR_train_second_all.csv'.format(data_name), dtype='int')
    # prob_test_info_0 = np.loadtxt(r'F:\codes-KBS\MR_test_second_all.csv'.format(data_name), dtype='int')
    # prob_train_info = np.append(prob_train_info_0, prob_train_info, axis=1)
    # prob_test_info = np.append(prob_test_info_0, prob_test_info, axis=1)
    # all_data = np.concatenate((prob_train_info, prob_test_info))
    # new_add_feature_2_id_2_list = np.concatenate((prob_train_info_0, prob_test_info_0))
    # # all_data = np.append(prob_train_info.reshape(8534,1), prob_test_info.reshape(1050,1),axis=1)
    # knn_simi_opp0 = np.append(my_binary_relation, knn_1, axis=0)
    # knn_simi_opp0 = np.append(my_binary, knn_1, axis=0)
    # knn_simi_opp0 = np.append(knn_simi_opp0, learning_interpretable_metric, axis=0 )
    # knn_simi_opp0 = learning_interpretable_metric
    # width = all_data_test.shape[1]

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
    # positive_num = [1,2,4,6,8,11,12,15,16,18]
    # all_data = np.concatenate((prob_train_info, prob_test_info), axis=0)
    # for num in range(all_data.shape[1]):
    #     line = dict()
    #     line['feature_id'] = num
    #     line['feature_type'] = 'unary_feature'
    #     line['parameterize'] = 1
    #     line['feature_name'] = 'dnn_'+str(num)
    #     dic_specific = dict()
    #     for each_num in range(len(all_data)):
    #         if all_data[each_num][num] == 100 or all_data[each_num][num]==-1 or all_data[each_num][num]==0:
    #             continue
    #         dic_specific[each_num] = [0.5 if num in positive_num else -0.5,  all_data[each_num][num]]
    #         # dic_specific[each_num] = [0.5 if all_data[each_num][num]==1  else -0.5, all_data[each_num][num]]
    #         # if num == 16:
    #         #     dic_specific[each_num]=[0.5 if all_data[each_num][num]==1 else -0.5, all_data[each_num][num]]
    #         # if num in[14, 15]:
    #         #     dic_specific[each_num] = [0.5, all_data[each_num][num]]
    #         if each_num not in var_id_2_feature_ids:
    #             var_id_2_feature_ids[each_num] = list()
    #         var_id_2_feature_ids[each_num].append(line['feature_id'])
    #     line['weight'] = dic_specific
    #     feature_id_2_weight[line['feature_id']] = dic_specific
    #     all_lines.append(line)

    # feature_start = 14
    # feature_start = 0
    # for line_i in new_add_feature_2_id_2_list:
    #     line_i_dic = dict()
    #     line_i_dic['feature_id'] = feature_start
    #     line_i_dic['feature_type'] = 'unary_feature'
    #     line_i_dic['parameterize'] = 0
    #     line_i_dic['feature_name'] = 'factor_{}'.format(line_i[0])
    #     dic_specific = dict()
    #     for each_num in eval(line_i[1]):
    #         dic_specific[each_num] = [0.5, 1]
    #         if each_num not in var_id_2_feature_ids:
    #             var_id_2_feature_ids[each_num] = list()
    #         var_id_2_feature_ids[each_num].append(line_i_dic['feature_id'])
    #     line_i_dic['weight'] = dic_specific
    #     if feature_start in feature_id_2_weight:
    #         assert  0
    #     feature_id_2_weight[line_i_dic['feature_id']] = dic_specific
    #     feature_start += 1
    #     all_lines.append(line_i_dic)


    # ## knn因子



    # knn_3_np = np.append(knn_3, knn_3_test, axis=0)
    # knn_neighors = [knn_1_np,  knn_3_np]
    # for index, knn in enumerate(knn_neighors):
    #     for row in range(knn.shape[0]):
    #         line_num = knn_1_np[row, :].astype(int)
    #         for i in range(1, len(line_num)):
    #             line = dict()
    #             line['feature_id'] = feature_start
    #             line['feature_type'] = 'binary_feature'
    #             line['parameterize'] = 0
    #             line['feature_name'] = 'dnn_{}'.format(index)
    #             dic_specific = dict()
    #             dic_specific[(line_num[0], line_num[i])] = [0.5, 1]
    #             if (line_num[0], line_num[i]) not in var_id_2_feature_ids_2:
    #                 var_id_2_feature_ids_2[(line_num[0], line_num[i])] = list()
    #             var_id_2_feature_ids_2[(line_num[0], line_num[i])].append(line['feature_id'])
    #             if line_num[0] not in var_id_2_feature_ids_2_list:
    #                 var_id_2_feature_ids_2_list[line_num[0]] = list()
    #             if line_num[i] not in var_id_2_feature_ids_2_list:
    #                 var_id_2_feature_ids_2_list[line_num[i]] = list()
    #             var_id_2_feature_ids_2_list[line_num[0]].append(line['feature_id'])
    #             var_id_2_feature_ids_2_list[line_num[i]].append(line['feature_id'])
    #             if line['feature_id'] not in feature_id_2_ids:
    #                 feature_id_2_ids[line['feature_id']] = list()
    #             feature_id_2_ids[line['feature_id']].append((line_num[0],line_num[1]))
    #             line['weight'] = dic_specific
    #             feature_id_2_weight[line['feature_id']] = dic_specific
    #             all_lines.append(line)
    #             feature_start += 1

    if data_name_1 == TYPE:
        len_train = len(embedding_train)
        # len_dev = len(embedding_dev)
        len_test = len(embedding_test)
    # elif data_name_1 == '5%-MR-第7次修正unlabeled':
    #     len_train = 2262 #2262
    #     len_dev = 754
    #     len_test = 754
    elif data_name_1 == '5%-MR-第7次修正unlabeled':
        len_train = 6228 # 2262        len_dev = 872
        len_test = 6228
    elif data_name_1 == '5%-MR-修正完u1后3%-CR-special':
        len_train = 2607 #2262
        len_dev = 754
        len_test = 377
    elif data_name_1 == 'MR获取u1':
        len_train = 2262 #2262
        len_dev = 754
        len_test = 755
    elif data_name_1 == '1%-MR-gml修正U1':
        len_train = 9062 #2262
        len_dev = 1066
        len_test = 534
    elif data_name_1 == '1%-MR-right-now-large':
        len_train = 8528 #2262
        len_dev = 1066
        len_test = 1067
    # elif data_name_1 == '5%-MR-第7次修正unlabeled':
    #     len_train = 9062 #2262
    #     len_dev = 1066
    #     len_test = 533
    elif data_name_1 == '1%-MR':
        len_train = 22 #2262
        len_dev = 754
        len_test = 755
    elif data_name_1 == '11%-MR':
        len_train = 1280 #2262
        len_dev = 1066
        len_test = 1067
    elif data_name_1 == 'MR-gml修正u4':
        len_train = 533#2262
        len_dev = 915
        len_test = 533
    elif data_name_1 == '20%-MR':
        len_train = 1706 #2262
        len_dev = 1078
        len_test = 1050
    elif data_name_1 == 'MR':
        len_train = 5098 #2550 #1530 #1020 #5098 4821
        len_dev = 915
        len_test = 2034
    elif data_name_1 == '1%-MR-filter':
        len_train = 4821 #2550 #1530 #1020 #5098 4821
        len_dev = 915
        len_test = 2034
    elif data_name_1 == 'MR-dnn修正u1-8-3':
        len_train = 509 #2550 #1530 #1020 #5098
        len_dev = 915
        len_test = 509
    elif data_name_1 == '1%-MR':
        len_train = 5098  #456 #2550 #1530 #1020 #5098
        len_dev = 916
        len_test = 2034
    # elif data_name_1 == '5%-MR-第7次修正unlabeled':
    #     len_train = 255  #456 #2550 #1530 #1020 #5098
    #     len_dev = 915
    #     len_test = 6877
    elif data_name_1 == '5%-MR':
        len_train = 509 #2550 #1530 #1020 #5098
        len_dev = 915
        len_test = 2034
    elif data_name_1 == "1%-MR-right-now":
        len_train = 8527 #8528
        len_dev =1066 # 1066
        len_test =1067   # 1067
    elif data_name_1 == "5%-MR-right-now-8-6":
        len_train = 2262#8528
        len_dev =754 # 1066
        len_test =755   # 1067
    elif data_name_1 == 'twitter2016':
        len_train = 3957
        len_dev = 1234
        len_test = 10290
    elif data_name_1 == "MR" or data_name_1 == "MR":
        len_train = 6920 #6920
        len_dev =872
        len_test =1821
    elif data_name_1 == "1%-MR-gml修正unlabeled":
        len_train = 6920 #6920
        len_dev =872
        len_test =6573
    elif data_name_1 == "1%-MR-gml预测u2":
        len_train = 7831 #6920
        len_dev =872
        len_test = 910
    elif data_name_1 == "2.1%-MR-first":
        len_train = 255 #6920
        len_dev =915
        len_test = 6877
    elif data_name_1 == "2.1%-MR-second":
        len_train = 255 #6920
        len_dev =915
        len_test = 6877
    # elif data_name_1 == "3%-MR-使用unlabeled不修正":
    #     len_train = 347 #6920
    #     len_dev =872
    #     len_test = 8394
    elif data_name_1 == "5%-MR":
        len_train = 692 #6920
        len_dev =872
        len_test =1821
    elif data_name_1 == "11%-MR":
        len_train = 1848 #6920
        len_dev =872
        len_test =1821
    elif data_name_1 == "20%-MR":
        len_train = 1384 #6920
        len_dev =872
        len_test =1821
    elif data_name_1 == "22%-MR":
        len_train = 1523 #6920
        len_dev =872
        len_test =1821
    elif data_name_1 == '2%-MR':
        len_train = 170 #2262
        len_dev = 1066
        len_test = 1067
    elif data_name_1 == 'MR-加入u1-预测u2':
        len_train = 5504 #2262
        len_dev = 915
        len_test = 1627
    elif data_name_1 == 'MR-加入u2-预测u3':
        len_train = 5830 #2262
        len_dev = 915
        len_test = 1301
    elif data_name_1 == 'MR-加入u3-预测u4':
        len_train = 6481  # 2262
        len_dev = 915
        len_test = 650
    elif data_name_1 == '1%-MR-使用修正后的unlabeled的数据':
        len_train = 6920  # 2262
        len_dev = 872
        len_test = 1821
    # elif data_name_1 == '5%-MR-第7次修正unlabeled':
    #     len_train = 114  # 2262
    #     len_dev = 754
    #     len_test = 2903
    elif data_name_1 == 'MR-gml对u2修正':
        len_train = 458  # 2262
        len_dev = 915
        len_test = 458
    elif data_name_1 == 'MR-对unlabeled-train+u1后的结果':
        len_train = 5607  # 2262
        len_dev = 915
        len_test = 1525
    elif data_name_1 == '1%-MR-使用修正后的unlabeled的数据-u3':
        len_train = 8285  # 2262
        len_dev = 872
        len_test = 456
    # #
    first_kind = []
    second_kind = []
    third_kind = []
    fourth_kind = []
    fifth_kind = []
    sixth_kind = []
    seventh_kind = []
    eighth_kind = []
    # for index, (edge0, edge1, value) in enumerate(my_binary_relation):
    #     edge0 = int(edge0)
    #     edge1 = int(edge1)
    #     if edge0 >=len_train and edge0<len_train+len_dev:
    #         print('??????????')
    #     elif edge0 >= len_train + len_dev:
    #         edge0 = edge0 - len_dev
    #
    #     if edge1 >=len_train and edge1<len_train+len_dev:
    #         print('??????????')
    #     elif edge1 >= len_train + len_dev:
    #         edge1 = edge1 - len_dev
    #     if edge0 > len_train+len_test:
    #         print("error!")
    #     if edge1 >len_train+ len_test:
    #         print('error!')
    #     knn_simi_opp_tmp.append((edge0, edge1, value))
    #     if value == 0:
    #         first_kind.append((edge0, edge1, value))  # oppose
    #     elif value ==1:
    #         second_kind.append((edge0, edge1, value))  # simliar


    # for index, (edge0, edge1, value) in enumerate(knn_simi_opp_tmp):
    #     if edge0 > len_train + len_test:
    #         print("error!")
    #     if edge1 > len_train + len_test:
    #         print('error!')
    # knn_simi_opp_tmp = after_filter(knn_simi_opp_tmp, all_edges)
    # knn_simi_opp = np.append(knn_simi_opp_tmp, all_edges, axis=0)
    # knn_simi_opp = knn_simi_opp_tmp
    for (edge0, edge1, value) in all_edges:
        # first_kind.append((int(edge0), int(edge1), value))
        first_kind.append((int(edge0), int(edge1), 1))

        # first_kind.append((int(edge0), int(edge1), 1))
        # first_kind.append((int(edge0), int(edge1), 1))
    # for (edge0, edge1, value) in all_edges1:
    #     second_kind.append((edge0, edge1, value))
    # # first_kind.extend(second_kind)
    # for (edge0, edge1, value) in all_edges2:
    #     third_kind.append((edge0, edge1, value))
    # first_kind.extend(third_kind)
    # for (edge0, edge1, value) in all_edges3:
    #     fourth_kind.append((edge0, edge1, value))
    # for (edge0, edge1, value) in all_edges4:
    #     fifth_kind.append((edge0, edge1, value))
    # for (edge0, edge1, value) in all_edges5:
    #     sixth_kind.append((edge0, edge1, value))
    # for (edge0, edge1, value) in all_edges6:
    #     seventh_kind.append((edge0, edge1, value))
    # for (edge0, edge1, value) in all_edges7:
    #     eighth_kind.append((edge0, edge1, value))



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
            # elif index == 4:
            # line['feature_name'] = 'fifth_simi'
        # elif index == 5:
        #     line['feature_name'] = 'sixth_simi'
        # else:
        #     line['feature_name'] = 'seventh_simi'
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
        # for (v1,v2) in dic_specific.keys():
            # if feature_start in feature_id_2_weight:
        #     assert  0
        feature_id_2_weight[line['feature_id']] = dic_specific

        all_lines.append(line)
        # feature_start += 1

    fw = open('./{}/only_one_'.format(data_name_1) + data_name_1 + "_features.pkl", 'wb')
    pickle.dump(all_lines, fw)

    return var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids

def generate_variable_pickl_only_has_factor(data_name_1, data_name, var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids):

    # test_info = pd.read_csv(r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\{}\test.csv".format(data_name), sep='\t',quoting=csv.QUOTE_NONE)
    # train_info = pd.read_csv(r"G:\transformers-4.3.3\res_small_data\input_data-3-1-10\{}\train.csv".format(data_name), sep='\t',quoting=csv.QUOTE_NONE)
    # test_info = pd.read_csv(r"E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\5%-MR-right-now\test.csv".format(data_name), sep='\t',quoting=csv.QUOTE_NONE)
    # train_info = pd.read_csv(r"E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\5%-MR-right-now\train.csv".format(data_name), sep='\t',quoting=csv.QUOTE_NONE)
    # train_info = pd.read_csv(r"E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\{}\train.csv".format(data_name), sep='\t',quoting=csv.QUOTE_NONE)
    # test_info = pd.read_csv(r"E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\1%-MR-right-now\test.csv".format(data_name), sep='\t',quoting=csv.QUOTE_NONE)
    # train_info = pd.read_csv(r"E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3-1-10\1%-MR-right-now\train.csv".format(data_name), sep='\t',quoting=csv.QUOTE_NONE)
    # type='new-add-test'
    type='special'
    test_info = pd.read_csv(embedding_csv_prefix + r"test.csv",sep='\t',quoting=csv.QUOTE_NONE)
    train_info = pd.read_csv(embedding_csv_prefix + r"train.csv",sep='\t',quoting=csv.QUOTE_NONE)
    # special_cases = {2300:1, 2465:0, 2479:0, 2669:1, 2900:0, 2970:1}

    easy_ids = list(train_info['id'])
    all_ids = list(train_info['id'])+list(test_info['id'])
    all_labels = list(train_info['label'])+list(test_info['label'])
    all_lines = list()
    easy_len = len(easy_ids)
    for elem in range(len(all_ids)):
        line = dict()
        var_id_val = elem
        # if var_id_val == 83:
        #     print('*********')
        line['var_id'] = var_id_val
        if var_id_val<easy_len:
            line['is_easy'] = True
        else:
            line['is_easy'] = False
        # if var_id_val in special_cases:
        #     line['is_easy'] = True
        #     line['label'] = special_cases[var_id_val]
        # else:

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

    fw = open('./{}/only_one_'.format(data_name_1) + data_name_1 + "_variables.pkl", 'wb')
    pickle.dump(all_lines, fw)

if __name__ == "__main__":
    # data_name = "twitter2016-all"
    # data_name_1 = 'twitter2016'
    # data_name = "twitter2016-all"
    # data_name_1 = 'twitter2016'
    # data_name = "MR-8534"
    # data_name_1 = 'MR'
    # data_name = "MR-all"
    # data_name_1 = 'MR'
    knn_csv_prefix = r'G:\DNN提取规则\all_result\3%-MR-修正unlabeled数据'
    embedding_csv_prefix = r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data-3\3%-MR-special\3%-MR-修正unlabeled数据\\'
    data_name = data_name_1 = knn_csv_prefix.split("\\")[-1]
    TYPE = data_name
    segment_symbol= True
    var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids = generate_feature_pickl_only_has_factor(data_name,data_name_1,TYPE,segment_symbol)
    generate_variable_pickl_only_has_factor(data_name_1, data_name, var_id_2_feature_ids, feature_id_2_weight, var_id_2_feature_ids_2, var_id_2_feature_ids_2_list, feature_id_2_ids)



