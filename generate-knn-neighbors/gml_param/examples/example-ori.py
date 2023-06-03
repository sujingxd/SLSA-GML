import pickle
import time
import warnings

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))


from gml import  GML
from gml_utils import *



if __name__ == '__main__':
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    warnings.filterwarnings('ignore')  # 过滤掉warning输出
    begin_time = time.time()

    # 1.准备数据
    dataname = "3%-MR-修正unlabeled数据"
    dir = 'data'
    with open(dir + "/only_one_" + dataname + '_variables.pkl', 'rb') as v:
        variables = pickle.load(v)
    with open(dir + "/only_one_" + dataname + '_features.pkl', 'rb') as f:
        features = pickle.load(f)
    # variables, features = extraction_of_partial_variables()
    # variables =  get_dic_variables(variables)
    # features = get_dic_feature(features)
    # variables, features = extract_small_data()
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