
import sklearn.metrics.pairwise as pairwise
import pandas as pd
import numpy as np
import os,  math
import pickle
from tqdm import trange, tqdm
from get_knn_count import get_knn_count
from tool import get_id_and_labels
from gml_param.examples.example import perform_dnn
# Calculate the center of high dimension data
def get_center(data_list):
    center = []

    for d in range(len(data_list[0])):# d for dimension
        coordinate = 0

        for data in data_list:
            coordinate += float(data[d])

        coordinate /= len(data_list)
        center.append(coordinate)

    return center


def take_0(elem):
    return elem[0]


def point_min(list):
    min_index = list.index(min(list))

    for i in range(len(list)):

        if i == min_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


def eval_distance(data_set, labels, predictions, distances):
    # Evaluate prediction
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1

    acc = correct / len(labels)


    # Evaluate distance
    d_predictions = []

    for info in distances:
        # info = list(map(float, info))
        d_predictions.append(info.index(min(info)))

    d_correct = 0

    for i in range(len(labels)):
        if d_predictions[i] == labels[i]:
            d_correct += 1

    d_acc = d_correct / len(labels)


    # Calculate different wrong
    evaluation = []

    for i in range(len(labels)):
        evaluation.append([labels[i], predictions[i], d_predictions[i]])

    p_wrong = d_wrong = 0

    for data_labels in evaluation:

        if data_labels[1] != data_labels[2]:

            if data_labels[1] == data_labels[0]:
                d_wrong += 1
            elif data_labels[2] == data_labels[0]:
                p_wrong += 1

    print('Dataset: {}'.format(data_set))
    print('Acc, D_Acc, d_right_p_wrong, p_right_d_wrong')
    print('{:.2f}, {:.2f}, {}, {}\n'.format(acc , d_acc , p_wrong, d_wrong))

import dill
def load_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = dill.load(f)
    return embedding

# Calculate the distance to each class center of every data
def get_distance(num_class, csv_dir):
    pre_csv_dir = csv_dir.split("\\")[:-1]
    pre_csv_dir = "\\".join(pre_csv_dir)
    # Calculate the class centers by train data
    # coordinate_train = np.loadtxt(os.path.join(csv_dir, 'train_sentence_out.tsv'))
    # coordinate_train = load_embedding(os.path.join(csv_dir, 'train_sentence_out.pkl'))
    coordinate_train = np.loadtxt(r'E:\2021-12-15-ori\input_data\CR\roberta-base-out\train_sentence_out.tsv')


    # coordinate_train = np.loadtxt(os.path.join(csv_dir, 'train_features.csv'))
    label = pd.read_csv(os.path.join(pre_csv_dir, 'labels_train.csv'), header=None).to_numpy()

    train_cls = []

    for cls in range(num_class):
        coordinate_cls = []

        for i in range(len(label)):

            if label[i][0] == cls:
                coordinate_cls.append(coordinate_train[i]) #属于各个类的embedding

        train_cls.append(coordinate_cls) # 加入2个类中心的所有embedding

    centers = []

    for i in range(len(train_cls)):
        centers.append(get_center(train_cls[i])) #对每类得到相应类中心,

    # for data_set in ['train', 'dev', 'test']:
    for data_set in ['train', 'test']:
    # for data_set in ['train_more', 'val_less', 'test']:

        # Read coordinate and label of data
        # coordinates = pd.read_csv(os.path.join(csv_dir, 'CR_distribution_{}.csv'.format(data_set)),
        #                               header=None).to_numpy()
        # coordinates = load_embedding(os.path.join(csv_dir, '{}_sentence_out.pkl'.format(data_set)))
        coordinates = np.loadtxt(r'E:\2021-12-15-ori\input_data\CR\roberta-base-out\{}_sentence_out.tsv'.format(data_set))
        # coordinates = np.loadtxt(os.path.join(csv_dir, '{}_sentence_out.tsv'.format(data_set)))
        # coordinates = np.loadtxt(os.path.join(csv_dir, '{}_features.csv'.format(data_set)))
        labels = pd.read_csv(os.path.join(pre_csv_dir, 'labels_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
        paths = pd.read_csv(os.path.join(pre_csv_dir, 'paths_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
        predictions = pd.read_csv(os.path.join(pre_csv_dir, 'predictions_{}.csv'.format(data_set)), header=None).to_numpy().flatten()


        # Calculate the distance to each class center of every data
        distance_to_center = pairwise.pairwise_distances(coordinates, centers, metric="cosine").tolist()
        # distance_to_center = pairwise.pairwise_distances(coordinates, centers).tolist()
        pd.DataFrame(distance_to_center).to_csv(os.path.join(csv_dir, 'distance_{}.csv'.format(data_set)), header=None, index=None)
        import torch.nn.functional as F
        new_distance_to_center = np.divide(np.array(distance_to_center)[:, 0], np.array(distance_to_center)[:, 1])
        # 底下需要进行输出距离比值的

        # asign = lambda t: 1.0 / (1.0 + np.exp(-t))
        # new_distance_to_center = asign(new_distance_to_center)
        pd.DataFrame(new_distance_to_center).to_csv(os.path.join(csv_dir, 'distance_{}_divide.csv' \
                                                         .format(data_set)), header=None, index=None)



        # Evaluate distance
        eval_distance(data_set, labels, predictions, distance_to_center)


        # Insert path and label of data to csv
        for i in range(len(distance_to_center)):
            distance_to_center[i].insert(0, paths[i])
            distance_to_center[i].insert(1, labels[i])

        # distance_to_center.sort(key=take_0)

        exec('distance_to_center_{} = distance_to_center'.format(data_set))


    # Merge 3 csvs together
    distance_center_all = []

    for data_set in ['train', 'test']:
    # for data_set in ['train_more', 'val_less', 'test']:
        exec('distance_center_all.extend(distance_to_center_{})'.format(data_set))

    # CReate the header of csv
    header = ['data', 'label']

    for i in range(num_class):
        header.append('class_{:0>3d}_distance'.format(i))


    # Save the final csv
    distance_center_all.insert(0, header)
    pd.DataFrame(distance_center_all).to_csv(os.path.join(csv_dir, 'all_distance.csv'\
                                                          ), header=None, index=None)

    # print('{}_{}_distance_to_center_all.csv saved at {}\n'.format(elem_name, layer, csv_dir))

    total_num = len(distance_center_all)
    # # Get nearest class csv
    # n_distance_center_all = []
    #
    # for i in range(len(distance_center_all) - 1):
    #     info = distance_center_all[i + 1]
    #     raw_distance = info[2:]
    #     n_info = [info[0], info[1]]
    #     n_info.extend(point_min(raw_distance))
    #     n_distance_center_all.append(n_info)
    #
    # n_header = ['data', 'label']
    #
    # for i in range(num_class):
    #     n_header.append('{}_{}_class_{:0>3d}_is_nearest'.format(elem_name, layer, i))
    #
    # n_distance_center_all.insert(0, n_header)
    #
    # pd.DataFrame(n_distance_center_all).to_csv(os.path.join(csv_dir, '{}_{}_min_distance.csv' \
    #                                                       .format(elem_name, layer)), header=None, index=None)

# settings


def merge_file(csv_dir, k_list):
    all = None
    file_distance = pd.read_csv(os.path.join(csv_dir, 'all_distance.csv')).values
    for k in k_list:
        knn_feature = pd.read_csv(os.path.join(csv_dir, 'knn{}.csv'.format(k))).values
        if all is None:
            all = knn_feature
        else:
            all = np.hstack((all, knn_feature[:, 2:]))
    all = np.hstack((all, file_distance[:, 2:]))
    column = all.shape[1]
    name_colum = ["factor_{}".format(i) for i in range(column-2)]
    name_colum.insert(0, "id")
    name_colum.insert(1, "label")
    pd.DataFrame(all, columns=name_colum).to_csv(csv_dir+ "\genera_pair_info_more.txt", index=None)



def merge_file_all(csv_dir, path_list):
    file_0 = pd.read_csv(os.path.join(path_list[0], 'genera_pair_info_more.txt')).values
    for i in range(1, len(path_list)):
        file_1 = pd.read_csv(os.path.join(path_list[i], 'genera_pair_info_more.txt')).values[:,2:]
        file_0 = np.hstack((file_0, file_1))
    column = file_0.shape[1]
    name_colum = ["factor_{}".format(i) for i in range(column-2)]
    name_colum.insert(0, "id")
    name_colum.insert(1, "label")
    pd.DataFrame(file_0, columns=name_colum).to_csv(csv_dir + r"\all_pair_info_more.csv", index=None)

def generate_pickle(path):
    dataset = ["train", "dev", "test"]
    for each_set in dataset:
        data_content = np.loadtxt(path+"\{}_sentence_out.txt".format(each_set))
        file_out = open(path+"\{}_sentence_out.pkl".format(each_set), "wb")
        pickle.dump(data_content, file_out)


if __name__ == '__main__':
    common_name = r"5%-twitter2016-special\不修正\使用unlabeled执行一遍gml"
    # common_name = r"0.5%-CR-special\修正400-最后"
    # common_name = r"1%-CR-special-12-28\修正unlabeled\使用unlabeled执行一遍gml"
    # distribution_csv_prefix = r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data-1\{}'.format(common_name)
    distribution_csv_prefix = r'E:\服务器上下载的\zcl\transformers-4.3.3\input_data\{}'.format(common_name)
    data_type = r'{}'.format(common_name)
    data_path = r'L:\gml-ER\sentilare_data'
    local_path = r"G:\DNN提取规则\all_result"
    new_genrate_path = local_path+"\\"+data_type.split('\\')[-1]
    segment_symbol =True
    if not os.path.exists(new_genrate_path):
        os.makedirs(new_genrate_path)
    data_names = ['twitter2016']
    fw = open(r'all_result\all_knn_acc_results.csv','w')
    name_line = 'data_name'+"\t"
    theta = 0
    data_files = list()
    for data_name in data_names:
        data_files.append(os.path.join(data_path, data_name))
    for csv_dir in data_files:
        num_class = 2
        k_list = [3]
        for i in k_list:
            name_line += "knn_"+str(i) + "\t"
        fw.write(name_line + "\n")
        dnn_type = ["bert-base-uncased"]
        path_list = [csv_dir+"\\"+dnn_type[i] for i in range(len(dnn_type))]
        for i in dnn_type:
            for CUR_NAME in data_names:
                path = csv_dir+"\\"+i
                get_id_and_labels(data_type, csv_dir)
                fw.write(CUR_NAME+"\t")
                get_knn_count(distribution_csv_prefix, CUR_NAME, k_list, num_class, csv_dir, path, i, fw, theta,new_genrate_path)
                fw.write(CUR_NAME + "\n")
    perform_dnn(common_name, segment_symbol)