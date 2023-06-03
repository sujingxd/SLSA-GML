
import sklearn.metrics.pairwise as pairwise
import pandas as pd
import numpy as np
import os,  math
import pickle
from tqdm import trange, tqdm
# from get_knn_count import get_knn_count
from get_knn_count_cluster import get_knn_count
from tool import get_id_and_labels
from sklearn.cluster import KMeans

n_clusters =10
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


def eval_distance(data_set, labels, predictions, distances, centers_2_label):
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
        d_predictions.append(centers_2_label[info.index(min(info))])

    d_correct = 0
    # pd.DataFrame(d_predictions).to_csv("d_prediction.csv",index=None,header=None)
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
    print('{:.2f}, {:.2f}, {}, {}\n'.format(acc * 100, d_acc * 100, p_wrong, d_wrong))


def load_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = pickle.load(f)
    return embedding

# Calculate the distance to each class center of every data
def get_distance(num_class, csv_dir):
    pre_csv_dir = csv_dir.split("\\")[:-1]
    pre_csv_dir = "\\".join(pre_csv_dir)
    # Calculate the class centers by train data
    coordinate_train = load_embedding(os.path.join(csv_dir, 'train_sentence_out.pkl'))
    label = pd.read_csv(os.path.join(pre_csv_dir, 'labels_train.csv'), header=None).to_numpy()
    temp_print =pairwise.pairwise_distances(coordinate_train, coordinate_train, metric="cosine").tolist()
    for data_set in ['train', 'dev', 'test']:
        coordinates = load_embedding(os.path.join(csv_dir, '{}_sentence_out.pkl'.format(data_set)))
        labels = pd.read_csv(os.path.join(pre_csv_dir, 'labels_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
        paths = pd.read_csv(os.path.join(pre_csv_dir, 'paths_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
        predictions = pd.read_csv(os.path.join(pre_csv_dir, 'predictions_{}.csv'.format(data_set)), header=None).to_numpy().flatten()
        # Calculate the distance to each class center of every data
        distance_to_center = pairwise.pairwise_distances(coordinates, centers, metric="cosine").tolist()
        positions = list()
        for i in distance_to_center:
            positions.append(i.index(min(i)))
        # pd.DataFrame(positions).to_csv("center_position.txt",header=None,index=None)

        pd.DataFrame(distance_to_center).to_csv(os.path.join(csv_dir, 'distance_{}.csv' \
                                                              .format(data_set)), header=None, index=None)
        # Evaluate distance

        eval_distance(data_set, labels, predictions, distance_to_center, centers_2_label)
        # Insert path and label of data to csv
        for i in range(len(distance_to_center)):
            distance_to_center[i].insert(0, paths[i])
            distance_to_center[i].insert(1, labels[i])
        # distance_to_center.sort(key=take_0)
        exec('distance_to_center_{} = distance_to_center'.format(data_set))
    # Merge 3 csvs together
    distance_center_all = []
    for data_set in ['train', 'dev', 'test']:
        exec('distance_center_all.extend(distance_to_center_{})'.format(data_set))
    # Create the header of csv
    header = ['data', 'label']
    for i in range(num_class):
        header.append('class_{:0>3d}_distance'.format(i))
    # Save the final csv
    distance_center_all.insert(0, header)
    pd.DataFrame(distance_center_all).to_csv(os.path.join(csv_dir, 'all_distance.csv'\
                                                          ), header=None, index=None)

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
        data_content = np.loadtxt(path+"\{}_sentence_out.tsv".format(each_set))
        file_out = open(path+"\{}_sentence_out.pkl".format(each_set), "wb")
        pickle.dump(data_content, file_out)

if __name__ == '__main__':
    data_path = r'G:\SentiLARE-master\input_data'
    data_names = ['CR']
    data_files = list()
    for data_name in data_names:
        data_files.append(os.path.join(data_path, data_name))
    for csv_dir in data_files:
        num_class = 2
        k_list = [1,2,3,4,5,6,7,8,9]
        # dnn_type = [ "xlnet-base-cased", "roberta-base",'deberta-base','mpnet-base']
        dnn_type = ["xlnet-base-cased", "roberta-base"]
        path_list = [csv_dir+"\\"+dnn_type[i]  for i in range(len(dnn_type))]
        for i in dnn_type:
            path = csv_dir+"\\"+i
            generate_pickle(path)
            get_id_and_labels(csv_dir)
            get_distance(num_class, path)
            CUR_NAME = "CR"
            get_knn_count(CUR_NAME, k_list, num_class, csv_dir, path, i)
            merge_file(path, k_list)
        merge_file_all(csv_dir, path_list)
    # num_class = 2
    # csv_dir ="G:\SentiLARE-master\input_data\CR\deberta-base"
    # get_distance(num_class, csv_dir)