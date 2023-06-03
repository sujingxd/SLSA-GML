import numpy as np
import pandas as pd


def interval(y):
    if y>0 and y <0.1:
        return 1
    elif y>= 0.1 and y <0.2:
        return 2
    elif y>= 0.2 and y <0.3:
        return 3
    elif y>= 0.3 and y <0.4:
        return 4
    elif y>= 0.4 and y <0.5:
        return 5
    elif y>= 0.5 and y <0.6:
        return 6
    elif y>= 0.6 and y <0.7:
        return 7
    elif y>= 0.7 and y <0.8:
        return 8
    elif y>= 0.8 and y <0.9:
        return 9
    elif y>= 0.9 and y <1:
        return 10

def get_common_factor():
    data_name = 'CR'
    data_set = 'train'
    n_clusters = 20
    topk = 5
    all_data = []
    distance_input_train = np.loadtxt('CR-all_distance_train_all_distance_20.csv')
    distance_input_test = np.loadtxt('CR-all_distance_test_all_distance_20.csv')
    distance_input = np.concatenate((distance_input_train, distance_input_test))
    feature_2_id = dict()
    for index, each_line in enumerate(distance_input):
        line_dic = dict()
        index_list = np.argsort(each_line)[0:topk]
        dis_list = each_line[index_list]
        line_dic['id'] = index
        line_dic['index_list'] = index_list
        line_dic['distance_list'] = dis_list
        line_dic['factor'] = [str(x)+"&"+str(interval(y)) for x, y in zip(index_list, dis_list)] #[7-2]
        all_data.append(line_dic['factor'])
        for each_data in line_dic['factor']:
            if each_data not in feature_2_id:
                feature_2_id[each_data] = list()
            else:
                feature_2_id[each_data].append(index)
    out = pd.DataFrame(all_data, columns=['name0', 'name1','name2','name3','name4'])
    out['id'] = list(range(len(distance_input)))
    out.to_csv('CR_new_add_factor.csv')
    out1 = pd.DataFrame()
    out1['feature_id'] = feature_2_id.keys()
    out1['id_list'] = feature_2_id.values()
    out1.to_csv('CR_feature_id.csv', index = None)



if __name__ == "__main__":
    get_common_factor()