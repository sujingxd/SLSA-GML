
import sklearn.metrics.pairwise as pairwise
import os
from os.path import join
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import pickle
from demo_distance import my_distance
from sklearn.metrics.pairwise import cosine_distances


def point_max(list):
    max_index = list.index(max(list))

    for i in range(len(list)):

        if i == max_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


def pairwise_distances_(distribution, distribution_train, metric="cosine"):
    total = []
    for index0, row0 in enumerate(distribution):
        row_np = []
        for index1, row1 in enumerate(distribution_train):
            dis = my_distance(row0, row1)
            row_np.append(dis)
        row_np = np.array(row_np)
        total.append(row_np)
    total = np.array(total)
    return total

#
# def get_knn(current_id, k, list, label,theta,train_len,type):
#     list = list.tolist()
#     list_sorted = sorted(list)
#     knn_labels = []
#     knn_distance = []
#     ids = []
#     i = 0
#     while len(knn_labels) < k:
#         index = list.index(list_sorted[i])
#         i += 1
#         if index == current_id + train_len and type == 'test':
#             continue
#         if type == 'train' and current_id == index:
#             continue
#         knn_labels.append(label[index])
#         knn_distance.append(list[index])
#         ids.append(index)
#     return knn_labels, ids, knn_distance


def get_knn(current_id, k, list, label,theta,train_len,type):
    list = list.tolist()
    list_sorted = sorted(list)
    knn_labels = []
    knn_distance = []
    ids = []
    i = 0
    total_train_cnt =4
    train_cnt = 0
    while len(knn_labels) < k:
        index = list.index(list_sorted[i])
        i += 1
        if index == current_id + train_len and type == 'test':
            continue
        if type == 'train' and current_id == index:
            continue
        if index < train_len and type == 'test':
            train_cnt += 1
        if train_cnt>4:
            break
        if index not in ids:
            knn_labels.append(label[index])
            knn_distance.append(list[index])
            ids.append(index)
    return knn_labels, ids, knn_distance


def eval_knn(data_set, labels, predictions, count):
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1

    acc = correct / len(labels)


    # Evaluate distance
    k_predictions = []

    for info in count:
        # info = list(map(int, info))
        k_predictions.append(info.index(max(info))) # 获取最多的标签为准确率。
    pd.DataFrame(k_predictions,columns=["k_pre"],index=None).to_csv("k_pre-1.csv")
    k_correct = 0

    for i in range(len(labels)):
        if i == 9:
            print("**********")
        if k_predictions[i] == labels[i]:
            k_correct += 1

    k_acc = k_correct / len(labels)


    # Calculate different wrong
    evaluation = []

    for i in range(len(labels)):
        evaluation.append([labels[i], predictions[i], k_predictions[i]])

    p_wrong = k_wrong = 0

    for data_labels in evaluation:

        if data_labels[1] != data_labels[2]:

            if data_labels[1] == data_labels[0]:
                k_wrong += 1
            elif data_labels[2] == data_labels[0]:
                p_wrong += 1

    print('Dataset: {}'.format(data_set))
    print('Acc, K_Acc, k_right_p_wrong, k_wrong_p_right')
    print('{:.2f}, {:.6f}, {}, {}\n'.format(acc * 100, k_acc * 100, p_wrong, k_wrong))


def load_embedding(file_path):
    with open(file_path, 'rb') as f:
          embedding = pickle.load(f)
    return embedding.astype(np.float32)

def get_edge_acc(all_lines,label_dic, fw):
    true_label = []
    for line in all_lines:
        src, tgr = line
        true_label.append(1 if label_dic.get(src)[0] == label_dic.get(tgr)[0] else 0)
    fw.write(str(sum(true_label)/len(true_label)))
    print(str(sum(true_label)/len(true_label)))
    fw.write("\t")

def get_all_neighors(all_lines, k, label_dic,fw):
    all_neighbors = []
    for i in range(len(all_lines)):
        src, _neighbors = all_lines[i]
        for each_nei in _neighbors:
            all_neighbors.append([src, each_nei])
    get_edge_acc(all_neighbors, label_dic, fw)
    return all_neighbors

def compute_binary_acc(knn_ids_np,  k, labels_current, labels_train, fw):
    labels_train = labels_train.reshape(len(labels_train),1)
    labels_current = labels_current.reshape(len(labels_current), 1)
    df = np.append(labels_train, labels_current,axis=0)
    ids = list(range(len(labels_train) + len(labels_current)))
    label = df
    label_dic = {ids[i]: label[i] for i in range(len(label))}
    get_all_neighors(knn_ids_np, k, label_dic, fw)


def get_data(distribution_train):
    new_distribution_train = []
    for i in range(len(distribution_train)):
        if i %2 ==0:
            new_distribution_train.append(distribution_train[i])
    return np.array(new_distribution_train)

# Calculate the knn_count to each class center of every data
def get_knn_count(distribution_csv_prefix, CUR_NAME, k_list, num_class, csv_dir, path, data_type, fw, theta, new_genrate_path):
    # distribution_train = load_embedding(os.path.join(path, 'train_sentence_out.pkl'))
    # distribution_dev = load_embedding(os.path.join(path, 'dev_sentence_out.pkl'))
    # distribution_test = load_embedding(os.path.join(path, 'test_sentence_out.pkl'))
    # train_temp = pd.read_csv('E:\stc_clustering-master\data\CR\stc_CR_new.csv').values[:2262]
    # dev_temp = pd.read_csv('E:\stc_clustering-master\data\CR\stc_CR_new.csv').values[2262:3016]
    # test_temp = pd.read_csv('E:\stc_clustering-master\data\CR\stc_CR_new.csv').values[3016:]

    # train_temp = np.loadtxt(r'E:\gml-old-平台-0\data\{}\feature_train.csv'.format(CUR_NAME))
    # dev_temp = np.loadtxt(r'E:\gml-old-平台-0\data\{}\feature_dev.csv'.format(CUR_NAME))
    # test_temp = np.loadtxt(r'E:\gml-old-平台-0\data\{}\feature_test.csv'.format(CUR_NAME))


    # distribution_train = np.loadtxt(r'F:\RoBERTaABSA-main\train_sentence_out.txt'.format(CUR_NAME))
    # distribution_dev =np.loadtxt(r'F:\RoBERTaABSA-main\dev_sentence_out.txt'.format(CUR_NAME))
    # distribution_test =np.loadtxt(r'F:\RoBERTaABSA-main\test_sentence_out.txt'.format(CUR_NAME))


    # distribution_train = np.loadtxt(r'E:\paddle-yuanshi\PaddleNLP-develop\train_{}_sentence_out1.csv'.format(CUR_NAME,CUR_NAME))
    # distribution_dev = np.loadtxt(r'E:\paddle-yuanshi\PaddleNLP-develop\dev_{}_sentence_out1.csv'.format(CUR_NAME,CUR_NAME))
    # distribution_test = np.loadtxt(r'E:\paddle-yuanshi\PaddleNLP-develop\test_{}_sentence_out1.csv'.format(CUR_NAME,CUR_NAME))

    # distribution_train = np.loadtxt(r'E:\codes-KBS\{}_train_t_sentence_all.csv'.format(CUR_NAME))
    # distribution_dev = np.loadtxt(r'E:\codes-KBS\{}_dev_t_sentence_all.csv'.format(CUR_NAME))
    # distribution_test = np.loadtxt(r'E:\codes-KBS\{}_test_t_sentence_all.csv'.format(CUR_NAME))
    # distribution_train = np.loadtxt(r'F:\sj_supattern_net\{}_train_t_sentence_all.csv'.format(CUR_NAME,CUR_NAME))
    # distribution_dev = np.loadtxt(r'F:\sj_supattern_net\{}_dev_t_sentence_all.csv'.format(CUR_NAME,CUR_NAME))
    # distribution_test = np.loadtxt(r'F:\sj_supattern_net\{}_test_t_sentence_all.csv'.format(CUR_NAME,CUR_NAME))
    #
    # distribution_train = pd.read_csv(r'F:\sj_supattern_net\{}_train_prob_out-all.csv'.format(CUR_NAME,CUR_NAME))['prob']
    # distribution_dev = pd.read_csv(r'F:\sj_supattern_net\{}_dev_prob_out-all.csv'.format(CUR_NAME,CUR_NAME))['prob']
    # distribution_test = pd.read_csv(r'F:\sj_supattern_net\{}_test_prob_out-all.csv'.format(CUR_NAME,CUR_NAME))['prob']


    # train_temp = np.loadtxt(r'F:\RoBERTaABSA-main\CR_train_sentence_out.txt'.format(CUR_NAME))
    # dev_temp = np.loadtxt(r'F:\RoBERTaABSA-main\CR_dev_sentence_out.txt'.format(CUR_NAME))
    # test_temp = np.loadtxt(r'F:\RoBERTaABSA-main\CR_test_sentence_out.txt'.format(CUR_NAME))


    # train_temp = np.loadtxt(r'F:\RoBERTaABSA-main\Dataset\CR\CR_train_sentence_out.txt'.format(CUR_NAME))
    # dev_temp = np.loadtxt(r'F:\RoBERTaABSA-main\Dataset\CR\CR_dev_sentence_out.txt'.format(CUR_NAME))
    # test_temp = np.loadtxt(r'F:\RoBERTaABSA-main\Dataset\CR\CR_test_sentence_out.txt'.format(CUR_NAME))


    # distribution_train = np.loadtxt(r'E:\按照数据集分割数据\CR\05\train_CR_sentence_out0.csv')
    # distribution_dev = np.loadtxt(r'E:\按照数据集分割数据\CR\05\dev_CR_sentence_out0.csv')
    # distribution_test = np.loadtxt(r'E:\按照数据集分割数据\CR\05\test_CR_sentence_out0.csv')

    # distribution_train = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\CR\11-22\train_sentence_out.tsv')
    # distribution_dev = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\CR\11-22\dev_sentence_out.tsv')
    # distribution_test = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\CR\11-22\test_sentence_out.tsv')
    # real_len = len(pd.read_csv(distribution_csv_prefix + r'\train_prediction.csv', sep='\t'))
    # distribution_train = np.loadtxt(distribution_csv_prefix + r'\train_sentence_out.tsv')[:real_len]
    distribution_train = np.loadtxt(distribution_csv_prefix + r'\train_sentence_out.tsv')
    # distribution_dev = np.loadtxt(distribution_csv_prefix + r'dev_sentence_out.tsv')
    distribution_test = np.loadtxt(distribution_csv_prefix + r'\test_sentence_out.tsv')
    #
    # distribution_train = np.loadtxt(r'F:\sj_supattern_net\{}_train_t_sentence_all.csv'.format(CUR_NAME))
    # distribution_dev = np.loadtxt(r'F:\sj_supattern_net\{}_dev_t_sentence_all.csv'.format(CUR_NAME))
    # distribution_test = np.loadtxt(r'F:\sj_supattern_net\{}_test_t_sentence_all.csv'.format(CUR_NAME))

    # distribution_train = np.loadtxt(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data\5%-CR-special\5%-CR-修正-U1-随机\test_sentence_out.tsv')
    # temp = distribution_train
    # train_pkl_file = open(r'G:\my_simcse\SimCSE-main\整理完后的文件-0.25%-CR-special-train.pkl', 'rb')
    # test_pkl_file = open(r'G:\my_simcse\SimCSE-main\整理完后的文件-0.25%-CR-special-test.pkl', 'rb')
    # dev_pkl_file = open(r'G:\my_simcse\SimCSE-main\整理完后的文件-0.25%-CR-special-dev.pkl', 'rb')
    # distribution_train = pickle.load(train_pkl_file)
    # distribution_test = pickle.load(test_pkl_file)
    # distribution_dev = pickle.load(dev_pkl_file)

    # train_pkl_file = open(r'G:\SimCSE-main\5%_CR_train.pkl', 'rb')
    # test_pkl_file = open(r'G:\SimCSE-main\5%_CR_test.pkl', 'rb')
    # distribution_train = pickle.load(train_pkl_file)
    # distribution_test = pickle.load(test_pkl_file)

    # distribution_train = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\单句子分类embedding\train_{}_sentence_out1.csv'.format(CUR_NAME,CUR_NAME))
    # distribution_dev = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\单句子分类embedding\dev_{}_sentence_out1.csv'.format(CUR_NAME,CUR_NAME))
    # distribution_test = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\单句子分类embedding\test_{}_sentence_out1.csv'.format(CUR_NAME,CUR_NAME))
    #
    # distribution_train = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\robert-large-sentence-embeding-single\train_sentence_out.tsv'.format(CUR_NAME))
    # distribution_dev = np.loadtxt( r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\robert-large-sentence-embeding-single\dev_sentence_out.tsv'.format(CUR_NAME))
    # distribution_test = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\robert-large-sentence-embeding-single\test_sentence_out.tsv'.format(CUR_NAME))



    # distribution_train = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\mnli-transformer-sentence-pre\train_sentence_out.tsv'.format(CUR_NAME))
    # distribution_dev = np.loadtxt( r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\mnli-transformer-sentence-pre\dev_sentence_out.tsv'.format(CUR_NAME))
    # distribution_test = np.loadtxt(r'L:\transformers-4.3.3-zcl-modify-loss\input_data\{}\mnli-transformer-sentence-pre\test_sentence_out.tsv'.format(CUR_NAME))

    # distribution_train = np.append(distribution_train, train_temp,axis=1)
    # distribution_dev = np.append(distribution_dev, dev_temp, axis=1)
    # distribution_test = np.append(distribution_test, test_temp, axis=1)

    # logits_train = np.loadtxt(os.path.join(path, 'train_logits_out.txt'))
    # logits_dev = np.loadtxt(os.path.join(path, 'dev_logits_out.txt'))
    # logits_test = np.loadtxt(os.path.join(path, 'test_logits_out.txt'))
    # distribution_train = np.append(distribution_train, logits_train,axis=1)
    # distribution_dev = np.append(distribution_dev, logits_dev, axis=1)
    # distribution_test = np.append(distribution_test, logits_test, axis=1)
    distributions = dict()
    distributions['train'] = distribution_train
    # distributions['dev'] = distribution_dev
    distributions['test'] = distribution_test
    need_only_test_flag = False
    # labels_train = pd.read_csv(join(csv_dir, 'labels_train.csv'), header=None).to_numpy(dtype='int').flatten()[:347]
    labels_train = pd.read_csv(join(csv_dir, 'labels_train.csv'), header=None).to_numpy(dtype='int').flatten()
    # labels_dev = pd.read_csv(join(csv_dir, 'labels_dev.csv'), header=None).to_numpy(dtype='int').flatten()
    labels_test = pd.read_csv(join(csv_dir, 'labels_test.csv'), header=None).to_numpy(dtype='int').flatten()
    # pre_probs= pd.read_csv(r'L:\gml-ER\sentilare_data\CR\bert-base-uncased\test_prediction.csv')['prob']
    # pre_labels = [1 if each_prob>=0.5 else 0 for each_prob in pre_probs]
    len_train = len(labels_train)
    # len_dev = len(labels_dev)
    # len_dev=1066
    # len_dev=1066
    # len_dev=754
    # len_dev=1066
    # len_dev = 754
    # len_dev =915
    # len_dev = 872
    len_test = len(labels_test)
    # dict_start_idx = {'train':0, 'dev':len_train, 'test': len_train+len_dev}
    distribution_train_test = np.append(distribution_train, distribution_test,axis=0)
    for k in k_list:
        count_all = []
        for data_set in ['train', 'test']:
            # labels_all = np.concatenate((labels_train, labels_dev, labels_test), axis=0)
            labels_all = np.concatenate((labels_train, labels_test), axis=0)
            if data_set == 'train':
                # paths = pd.read_csv(join(csv_dir, 'paths_{}.csv'.format(data_set)), header=None).to_numpy().flatten()[:347]
                # labels = pd.read_csv(join(csv_dir, 'labels_{}.csv'.format(data_set)), header=None).to_numpy().flatten()[:347]
                # predictions = pd.read_csv(join(csv_dir, 'predictions_{}.csv'.format(data_set)), header=None).to_numpy().flatten()[:347]

                paths = pd.read_csv(join(csv_dir, 'paths_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
                labels = pd.read_csv(join(csv_dir, 'labels_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
                predictions = pd.read_csv(join(csv_dir, 'predictions_{}.csv'.format(data_set)), header=None).to_numpy().flatten()

            else:
                paths = pd.read_csv(join(csv_dir, 'paths_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
                labels = pd.read_csv(join(csv_dir, 'labels_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
                # predictions = pd.read_csv(join(csv_dir, 'predictions_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
                # predictions = pd.read_csv(join(csv_dir, 'predictions_{}.csv'.format(data_set)),header=None).to_numpy().flatten()
                predictions = pd.read_csv(join(csv_dir, 'predictions_{}.csv'.format(data_set)),header=None).to_numpy().flatten()
            distribution = distributions[data_set]
            ###
            # dev_info = np.loadtxt(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data\CR\roberta-basedev_sentence_out.tsv')
            # train_info = np.loadtxt(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data\CR\roberta-basetrain_sentence_out.tsv')
            # test_info = np.loadtxt(r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data\CR\roberta-basetest_sentence_out.tsv')
            # train_info = [[-1,2,3],[2,3,2]]
            # test_info = [[1,-2,3],[2,3,2]]
            ###
            # pairwise_distances = pairwise.pairwise_distances(train_info, test_info, metric="cosine") #distance to train data
            # tmp = pairwise_distances
            pairwise_distances = pairwise.pairwise_distances(distribution, distribution_train_test, metric="cosine")
            if data_set == 'train':
                pairwise_distances_train_train = pairwise.pairwise_distances(distribution, distribution_train, metric="cosine")
                knn_train = []
                knn_ids_train = []
                knn_distance_train = []
                for i in trange(len(pairwise_distances_train_train)):
                    temp = get_knn(i, k, pairwise_distances_train_train[i], labels_train, theta, len_train,'train')
                    knn_train.append(temp[0])
                    knn_ids_train.append(temp[1])
                    knn_distance_train.append(temp[2])
                    first_ids = list(range(len_train))
                pdf = pd.DataFrame()
                pdf['id'] = first_ids
                pdf['neighbors'] = knn_ids_train
                pdf['distance'] = knn_distance_train
                pdf.to_csv(new_genrate_path+"\{}_knn_{}_of_{}_{}_train_train.txt".format(CUR_NAME, k, data_set, data_type), sep='\t',index=None)


            knn = []
            knn_ids = []
            knn_distance = []
            for i in trange(len(pairwise_distances)):
                temp = get_knn(i,  k, pairwise_distances[i], labels_all, theta,len_train,'test')
                knn.append(temp[0])
                knn_ids.append(temp[1])
                knn_distance.append(temp[2])
            if data_set == 'test':
                # knn_train_ids = np.array(knn_ids)
                first_ids = list(range(len_train, len_train+len_test))
                pdf = pd.DataFrame()
                pdf['id'] = first_ids
                pdf['neighbors'] = knn_ids
                pdf['distance'] = knn_distance
                # knn_ids_np = np.hstack((first_ids, knn_train_ids))
                # compute_binary_acc(pdv, k, labels, labels_train, fw)
                # np.savetxt("all_result\{}_knn_{}_of_{}_{}.txt".format(CUR_NAME, k, data_set, data_type), knn_ids_np, fmt="%s", delimiter="\t")
                pdf.to_csv(new_genrate_path + "\{}_knn_{}_of_{}_{}.txt".format(CUR_NAME, k, data_set, data_type), sep='\t',index=None)

            # elif data_set == 'train':
            #     # pairwise_distances_train_train = pairwise.pairwise_distances(distribution, distribution_train,metric="cosine")
            #     knn_train_ids = np.array(knn_ids)
            #     first_ids = np.array(list(range(len_train)))
            #     first_ids = first_ids.reshape(-1, 1)
            #     knn_ids_np = np.hstack((first_ids, knn_train_ids))
            #     # compute_binary_acc(knn_ids_np,  k, labels, labels_train,fw)
            #     np.savetxt("all_result\{}_knn_{}_of_{}_{}.txt".format(CUR_NAME, k, data_set, data_type), knn_ids_np, fmt="%s", delimiter="\t",index=None, header=None)
            elif data_set == "test" and need_only_test_flag:

                pairwise_distances_test = pairwise.pairwise_distances(distribution, distribution, metric="cosine")  # distance to dev data
                knn_test = []
                knn_ids_test = []
                for i in trange(len(pairwise_distances_test)):
                    temp_test = get_knn(i, k, pairwise_distances_test[i], labels_all, theta,len_train)
                    knn_test.append(temp_test[0])
                    knn_ids_test.append(temp_test[1])
                knn_train_ids = np.array(knn_ids)
                knn_test_ids = np.array(knn_ids_test)
                knn_ids_np = np.concatenate((knn_train_ids, knn_test_ids),axis=1)
                first_ids  = np.array(list(range(len_train+len_dev,len_train+len_dev+len_test)))
                first_ids = first_ids.reshape(-1,1)
                knn_ids_np = np.hstack((first_ids, knn_ids_np))
                # compute_binary_acc(knn_ids_np,  k, labels, labels_test,fw)
                pd.DataFrame(knn_ids_np, index=None).to_csv("{}_lnn_all_neighors_{}_of_{}_{}.csv".format(CUR_NAME, k, data_set,data_type),index=None)


            count = []
            for i in range(len(knn)):
                line = np.bincount(knn[i], minlength=num_class).tolist()
                count.append(line)


            # Evaluate knn_count
            print('===== K: {} ====='.format(k))
            eval_knn(data_set, labels, predictions, count)


            # Combine all knn_count
            for i in range(len(count)):
                count[i].insert(0, paths[i])
                count[i].insert(1, labels[i])

            count_all.extend(count)


        # CReate the header of csv
        header = ['data', 'label']

        for i in range(num_class):
            header.append('class_{:0>3d}_count{}'.format(i, k))


        # Save the final csv
        count_all.insert(0, header)

