
import sklearn.metrics.pairwise as pairwise
import pandas as pd
import numpy as np
import os,  math
import pickle
from tqdm import trange, tqdm
from get_knn_count import get_knn_count
from tool import get_id_and_labels
from sklearn.cluster import KMeans
import dill
from sklearn.cluster import SpectralClustering

def load_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = dill.load(f)
    return embedding


def filter_center(all_subparts, pre_csv_dir):
    label = np.array(pd.read_csv(os.path.join(pre_csv_dir, 'labels_train.csv'), header=None)[0])
    index_cluster = { ind:assign_label for ind, assign_label in enumerate(all_subparts)}
    clsuter_2_ids = {}
    filtered_cluster_2_ids = {}
    cluster_2_labels = {}
    pos_clusters = []
    neg_clusters = []
    for id, assi_label in index_cluster.items():
        if assi_label not in clsuter_2_ids:
            clsuter_2_ids[assi_label] = list()
        clsuter_2_ids[assi_label].append(id)
    for ass_label, ids in clsuter_2_ids.items():
        contained_labels = label[ids]
        cluster_2_labels[ass_label] = contained_labels
        if  len(contained_labels)== sum(contained_labels):
            filtered_cluster_2_ids[ass_label] = ids
            pos_clusters.append(ass_label)
        if sum(contained_labels)==0:
            neg_clusters.append(ass_label)
    return filtered_cluster_2_ids , cluster_2_labels, pos_clusters, neg_clusters

def softmax(X):
    return np.exp(X) / np.sum(np.exp(X))

def get_distance(num_class, csv_dir, n_clusters, data_name):
    pre_csv_dir = csv_dir.split("\\")[:-1]
    pre_csv_dir = "\\".join(pre_csv_dir)
    coordinate_train = np.loadtxt(os.path.join(csv_dir, 'train_sentence_out.tsv'))
    # add_vector_train = np.loadtxt(r'G:\transformers-4.3.3\all_matrix_train.txt')
    # coordinate_train = np.concatenate((coordinate_train, add_vector_train), axis=1)
    km = KMeans(n_clusters=n_clusters)
    km.fit(coordinate_train)
    all_contained_cluster = km.cluster_centers_
    all_subparts = km.labels_
    np.savetxt('assigned_labels.csv', all_subparts, fmt="%s", delimiter="\t")
    filtered_cluster_2_ids, _, pos_clusters, neg_clusters = filter_center(all_subparts, pre_csv_dir)

    for data_set in ['test']:
        coordinates = np.loadtxt(os.path.join(csv_dir, data_set+'_sentence_out.tsv'))
        # add_vector = np.loadtxt(r'G:\transformers-4.3.3\all_matrix_{}.txt'.format(data_set))
        # coordinates = np.concatenate((coordinates, add_vector), axis=1)
        train_data_to_distance_pos = None
        train_data_to_distance_neg = None
        all_minum_dis = []
        all_max_dis = []
        res_arr = []
        for ind, center_i in enumerate(all_contained_cluster):
            if ind in pos_clusters:
                center_i = center_i.reshape(1, -1)
                distance_to_center = pairwise.pairwise_distances(coordinates, center_i, metric="cosine")
                if train_data_to_distance_pos is None:
                    train_data_to_distance_pos = distance_to_center
                else:
                    train_data_to_distance_pos = np.hstack((train_data_to_distance_pos, distance_to_center))
            if ind in neg_clusters:
                center_i = center_i.reshape(1, -1)
                distance_to_center = pairwise.pairwise_distances(coordinates, center_i, metric="cosine")
                if train_data_to_distance_neg is None:
                    train_data_to_distance_neg = distance_to_center
                else:
                    train_data_to_distance_neg = np.hstack((train_data_to_distance_neg, distance_to_center))
        for each_line in train_data_to_distance_pos:
            each_line.sort()
            all_minum_dis.append(each_line[0])
        for each_line in train_data_to_distance_neg:
            each_line.sort()
            all_max_dis.append(each_line[0])
        for min_x, max_x in zip(all_minum_dis, all_max_dis):
            res_arr.append(min_x/max_x)
        # res = softmax(res_arr)
        np.savetxt('{}_distance_{}_all_distance_{}.csv'.format(data_name, data_set, n_clusters), res_arr, fmt="%s", delimiter="\t")




if __name__ == '__main__':
    # data_path = r'E:\transformers-4.3.3-zcl\input_data'
    data_path = r'G:\transformers-4.3.3\input_data'
    data_names = ['twitter2016']
    data_files = list()
    for data_name in data_names:
        data_files.append(os.path.join(data_path, data_name))
    for csv_dir in data_files:
        num_class = 2
        n_clusters = 20
        dnn_type = ["bert-base-uncased"]
        path_list = [csv_dir+"\\"+dnn_type[i]  for i in range(len(dnn_type))]
        for i in dnn_type:
            path = csv_dir+"\\"+i
            get_distance(num_class, path, n_clusters, data_names[0])