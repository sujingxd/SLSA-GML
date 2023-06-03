import random
from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import gml_utils
from pyds import MassFunction
import math
import pickle


class Regression:
    '''
    Calculate evidence support by linear regression
    '''
    def __init__(self, evidences, n_job, effective_training_count_threshold = 2,para=None,factor_type='unary'):
        '''
        @param evidences:
        @param n_job:
        @param effective_training_count_threshold:
        '''
        self.para = para
        self.effective_training_count = max(2, effective_training_count_threshold)
        self.n_job = n_job
        if len(evidences) > 0:
            XY = np.array(evidences)
            self.X = XY[:, 0].reshape(-1, 1)
            self.Y = XY[:, 1].reshape(-1, 1)
        else:
            self.X = np.array([]).reshape(-1, 1)
            self.Y = np.array([]).reshape(-1, 1)
        self.balance_weight_y0_count = 0
        self.balance_weight_y1_count = 0
        for y in self.Y:
            if y > 0:
                self.balance_weight_y1_count += 1
            else:
                self.balance_weight_y0_count += 1
        self.sample_weight_list = None
        if factor_type == 'unary':
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                self.sample_weight_list = list()
                adjust_coefficient = 0.7
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                sample_weight *= (adjust_coefficient)
                for y in self.Y:
                    if y[0] > 0:
                        self.sample_weight_list.append(sample_weight)
                    else:
                        self.sample_weight_list.append(1)
        elif factor_type == 'sliding_window':
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                self.sample_weight_list = list()
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                # blocking = 10
                # sample_weight = blocking
                for y in self.Y:
                    if y[0] > 0:
                        self.sample_weight_list.append(sample_weight)
                    else:
                        self.sample_weight_list.append(1)
                print('sliding_window Weight', sample_weight)
        # elif factor_type == 'binary':
        #     if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
        #         self.sample_weight_list = list()
        #         sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
        #         for y in self.Y:
        #             if y[0] > 0:
        #                 self.sample_weight_list.append(sample_weight)
        #             else:
        #                 self.sample_weight_list.append(1)
        #         print('binary Weight', sample_weight)
        self.perform()

    def perform(self):
        '''
        Perform linear regression
        @return:
        '''
        self.N = np.size(self.X)
        if self.N <= self.effective_training_count:
            self.regression = None
            self.residual = None
            self.meanX = None
            self.variance = None
            self.k = None
            self.b = None
        else:
            # sample_weight_list = None
            # if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
            #     sample_weight_list = list()
            #     sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
            #     blocking = 5
            #     sample_weight = blocking
            #     for y in self.Y:
            #         if y[0] > 0:
            #             sample_weight_list.append(sample_weight)
            #         else:
            #             sample_weight_list.append(1)
            self.Y = np.array(self.Y, dtype=np.float)
            # print('sample_weight_list: ',self.sample_weight_list)
            self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y, sample_weight=self.sample_weight_list)
            self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
            self.meanX = np.mean(self.X)  #The average value of feature_value of all evidence variables of this feature
            self.variance = np.sum((self.X - self.meanX) ** 2)
            self.k = self.regression.coef_[0][0]
            self.b = self.regression.intercept_[0]

class EvidentialSupport:
    '''
    Calculated evidence support
    '''
    def __init__(self,variables,features,varname2isevidence,vname2classname_fv,fv2probability):
        '''
        @param variables:
        @param features:
        '''
        vname2pairid_pkl = open('../GF/pkl/vname2pairid.pkl', 'rb')
        self.vname2pairid = pickle.load(vname2pairid_pkl)
        self.variables = variables
        self.features = features   #
        self.features_easys = dict()  # Store all easy feature values of all features    :feature_id:[[value1,bound],[value2,bound]...]
        self.tau_and_regression_bound = 10
        self.NOT_NONE_VALUE = 1e-8
        self.n_job = 10
        self.delta = 2
        self.effective_training_count_threshold = 2
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
        self.varname2isevidence = varname2isevidence
        self.vname2classname_fv = vname2classname_fv
        self.fv2probability = fv2probability
        self.observed_variables_set = set()
        self.poential_variables_set = set()
        self.decay = 1
        # edit duofenlei
        self.classNum = 2


    def get_unlabeled_var_feature_evi(self):
        '''
        Calculate the ratio of 0 and 1 in the evidence variable associated with each unary feature of each hidden variable,
        and the variable id at the other end of the binary feature,
        and finally add the two attributes of unary_feature_evi_prob and binary_feature_evi to each hidden variable
        @return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        unary_feature_evi_prob = list()
        binary_feature_evi = list()
        for id in self.poential_variables_set:
            for (feature_id,value) in self.variables[id]['feature_set'].items():
                feature_name = self.features[feature_id]['feature_name']
                weight = self.features[feature_id]['weight']
                if self.features[feature_id]['feature_type'] == 'unary_feature' and self.features[feature_id]['parameterize'] == 0:
                    labeled0_vars_num = 0
                    labeled1_vars_num = 0
                    for var_id in weight:
                        if self.variables[var_id]['is_evidence'] == True and self.variables[var_id]['label'] == 0:
                            labeled0_vars_num += 1
                        elif self.variables[var_id]['is_evidence'] == True and self.variables[var_id]['label'] == 1:
                            labeled1_vars_num += 1
                    if labeled1_vars_num==0 and labeled0_vars_num == 0:
                        continue
                    n_samples = labeled0_vars_num + labeled1_vars_num
                    neg_prob = float(labeled0_vars_num/n_samples)
                    pos_prob = float(labeled1_vars_num/n_samples)
                    unary_feature_evi_prob.append((feature_id,feature_name,n_samples,neg_prob,pos_prob))
                elif self.features[feature_id]['feature_type'] == 'binary_feature' and self.features[feature_id]['parameterize'] == 0 :
                    for (id1,id2) in weight:
                        anotherid = id1 if id1!=id else id2
                        if self.variables[anotherid]['is_evidence'] == True:
                            binary_feature_evi.append((anotherid,feature_name,feature_id))
            self.variables[id]['unary_feature_evi_prob'] = copy(unary_feature_evi_prob)
            self.variables[id]['binary_feature_evi'] = copy(binary_feature_evi)
            unary_feature_evi_prob.clear()
            binary_feature_evi.clear()

    def get_dict_rel_acc(self):
        '''
        get the accuracy of different types of relations
        :return:
        '''
        relations_name = set()
        relations_id = set()
        dict_rel_acc = dict()
        for feature in self.features:
            if feature['parameterize'] == 0:
                feature_id = feature['feature_id']
                if feature['feature_type'] == 'binary_feature':
                    relations_id.add(feature_id)
                    relations_name.add(feature['feature_name'])
        dict_reltype_edges = {rel: list() for rel in relations_name}
        for fid in relations_id:
            if feature['parameterize'] == 0:
                weight = self.features[fid]['weight']
                for (vid1,vid2) in weight:
                    if self.variables[vid1]['is_evidence'] == True and self.variables[vid2]['is_evidence'] == True:
                        dict_reltype_edges[self.features[fid]['feature_name']].append((vid1,vid2))
        for rel,edges  in dict_reltype_edges.items():
            ture_rel_num = 0
            if len(edges) > 0:
                for (vid1,vid2) in edges:
                        if 'simi' in rel and self.variables[vid1]['label'] == self.variables[vid2]['label']:
                            ture_rel_num +=1
                        elif 'oppo' in rel and self.variables[vid1]['label'] != self.variables[vid2]['label']:
                            ture_rel_num +=1
                        dict_rel_acc[rel] = float(ture_rel_num/len(edges))
            else:
                dict_rel_acc[rel] = 0.9
        return dict_rel_acc

    def construct_mass_function_for_propensity(self,uncertain_degree, label_prob, unlabel_prob):
        '''
        l: support for labeling
        u: support for unalbeling
        @param uncertain_degree:
        @param label_prob:
        @param unlabel_prob:
        @return:
        '''
        #The mass function needed for binary_factor to calculate evidence support
        return MassFunction({'l': (1 - uncertain_degree) * label_prob,
                             'u': (1 - uncertain_degree) * unlabel_prob,
                             'lu': uncertain_degree})

    def construct_mass_function_for_para_feature(self,theta):
        '''
        @param theta:
        @return:
        '''
        #The mass function needed for unary_factor to calculate evidence support
        return MassFunction ({'l':theta,'u':1-theta})

    def labeling_propensity_with_ds(self,mass_functions):
        combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=True)
        return combined_mass

    def computer_unary_feature_es(self,update_feature_set):
        '''
        Evidence support for calculating parameterized unary features
        @param update_feature_set:
        @return:
        '''
        data = list()
        row = list()
        col = list()
        var_len = 0
        fea_len = 0
        for fea in self.features:
            if fea['parameterize'] == 1:
                fea_len += 1
        for index, var in enumerate(self.variables):
            count = 0
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                if self.features[feature_id]['parameterize'] == 1:
                    count += 1
            if count > 0:
                var_len += 1
        for index, var in enumerate(self.variables):
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                if self.features[feature_id]['parameterize'] == 1:
                    data.append(feature_set[feature_id][1] + 1e-8)
                    row.append(index)
                    col.append(feature_id)
        self.data_matrix=csr_matrix((data, (row, col)), shape=(var_len, fea_len))
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.separate_feature_value()
        self.influence_modeling(update_feature_set)
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        #coefs = []
        #intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []
        for feature in self.features:
            if feature['parameterize'] == 1:
                if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >= 0:
                    feature['monotonicity'] = True
                else:
                    feature['monotonicity'] = False
                if feature['regression'].regression is not None and feature['monotonicity'] == True:
                    #coefs.append(feature['regression'].regression.coef_[0][0])
                    #intercept.append(feature['regression'].regression.intercept_[0])
                    zero_confidence.append(1)
                else:
                    #coefs.append(0)
                    #intercept.append(0)
                    zero_confidence.append(0)
                Ns.append(feature['regression'].N if feature['regression'].N > feature[
                    'regression'].effective_training_count else np.NaN)
                residuals.append(feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
                meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
                variance.append(feature['regression'].variance if feature['regression'].variance is not None else np.NaN)
        if len(zero_confidence) != 0:
            zero_confidence = np.array(zero_confidence)[col]
        if len(residuals) != 0 and len(Ns) != 0 and len(meanX) != 0 and len(variance) != 0:
            residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                         np.array(variance)[col]
            tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
            confidence = np.ones_like(data)
            confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
            confidence = confidence * zero_confidence
            evidential_support = (1 + confidence) / 2  # 正则化
            # 将计算出的evidential support写回
            csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(var_len, fea_len))
            for index, var in enumerate(self.variables):
                feature_set = self.variables[index]['feature_set']
                for feature_id in feature_set:
                    if self.features[feature_id]['parameterize'] == 1:
                        feature_set[feature_id][0] = csr_evidential_support[index, feature_id]

    def evidential_support(self,variable_set,update_feature_set):
        '''
        Evidence support for fusing all factors of each latent variable ,Only update the changed features each time
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        # Select different mass functions according to factor types and calculate evidence support
 #       self.dict_rel_acc = self.get_dict_rel_acc()
 #       self.get_unlabeled_var_feature_evi()
        mass_functions = list()
        self.computer_unary_feature_es(update_feature_set)
        for vid in variable_set:
            var = self.variables[vid]
            for fid in self.variables[vid]['feature_set']:
                if self.features[fid]['feature_type'] == 'unary_feature': #Judgment factor type
                    if self.features[fid]['parameterize']  == 1 : # Whether the factor is parameterized
                        mass_functions.append(self.construct_mass_function_for_para_feature(var['feature_set'][fid][0]))
                    else :
                        if 'unary_feature_evi_prob' in self.variables[vid]:
                            if len(var['unary_feature_evi_prob']) > 0:
                                for (feature_id, feature_name, n_samples, neg_prob, pos_prob) in var['unary_feature_evi_prob']:
                                    mass_functions.append(self.construct_mass_function_for_propensity(self.word_evi_uncer_degree, max(pos_prob, neg_prob),min(pos_prob, neg_prob)))
                if self.features[fid]['feature_type'] == 'binary_feature':
                    if 'binary_feature_evi' in var:
                        if len(var['binary_feature_evi']) > 0:
                            for (anotherid, feature_name, feature_id) in var['binary_feature_evi']:
                                rel_acc = self.dict_rel_acc[feature_name]
                                mass_functions.append(self.construct_mass_function_for_propensity(self.relation_evi_uncer_degree,rel_acc, 1-rel_acc))
            if len(mass_functions) > 0: #Calculate evidence support by D-S theory
                combine_evidential_support = self.labeling_propensity_with_ds(mass_functions)
                var['evidential_support'] = combine_evidential_support['l']
                mass_functions.clear()
            else:
                var['evidential_support'] = 0.0
        logging.info("evidential_support calculate finished")

    def separate_feature_value(self):
        '''
        Select the easy feature value of each feature for linear regression
        :return:
        '''
        # 多分类逻辑
        # evidences = list()
        # self.features_easys.clear()
        # for feature in self.features:
        #     if feature['parameterize'] == 1:
        #         evidences.clear()
        #         #  ==1 应该修改成等于当前因子的指示类  *10   edit  单因子回归均为正
        #         if feature['feature_type'] == 'unary_feature':
        #             for var_id, value in feature['weight'].items():
        #                 if var_id in self.observed_variables_set:
        #                     evidences.append([value[1], (1 if self.variables[var_id]['label'] == feature['Association_category'] else -1) * self.tau_and_regression_bound])
        #                     #evidences.append([value[1], (1 if self.variables[var_id]['label'] == self.variables[var_id]['label'] else -1) * self.tau_and_regression_bound])
        #         else:
        #             for var_ids, value in feature['weight'].items():
        #                 if var_ids[0] in self.observed_variables_set and var_ids[1] in self.observed_variables_set:
        #                     evidences.append([value[1], (1 if self.variables[var_ids[0]]['label'] == self.variables[var_ids[1]]['label'] else -1) * self.tau_and_regression_bound])
        #         self.features_easys[feature['feature_id']] = copy(evidences)
        # ourseed = 2022
        # random.seed(2022)
        binary_featureid = None
        each_feature_easys = list()
        self.features_easys.clear() # 滑动窗口easy直接放到 self.features_easys
        for feature in self.features:
            if feature['feature_name'] =='sliding_window_factor': # 滑动窗口easy直接放到sliding_window_feature_easys
                sliding_window_feature_easys=[]
                for var_id,value in feature['weight'].items():
                    if type(var_id) == int:
                        if var_id in self.observed_variables_set:
                            # sliding_window_feature_easys.append([value[1],((self.fv2probability[value[1]]-1) if abs(value[1]-0.0)<0.01 else self.fv2probability[value[1]])*20])
                            sliding_window_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
                print('sliding_window evidence: ',len(sliding_window_feature_easys))
                self.features_easys[1] = copy(sliding_window_feature_easys)
            if feature['parameterize'] == 1 and feature['feature_name']!='sliding_window_factor':
                each_feature_easys.clear()  # 不是滑动窗口easy的直接放到each_feature_easys
                for var_id, value in feature['weight'].items():
                    if type(var_id) ==int: # 单因子
                        if var_id in self.observed_variables_set:
                            each_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
                    if type(var_id) ==tuple: # 双因子
                        binary_featureid = feature['feature_id']
                        if var_id[0] in self.observed_variables_set and var_id[1] in self.observed_variables_set:
                            if self.variables[var_id[0]]['label'] == self.variables[var_id[1]]['label']:
                                each_feature_easys.append([value[1], 1*self.tau_and_regression_bound])
                            else :
                                each_feature_easys.append([value[1], -1*self.tau_and_regression_bound])
                            # print('双因子：',len(each_feature_easys))
                # if feature['feature_name']=='unary_rank_factor':
                #     np.random.seed(2022)
                #     sampleNO = np.random.choice(len(each_feature_easys), int(len(each_feature_easys)*self.decay), replace=False)
                #     print('self.decay',self.decay)
                #     aftersample = [each_feature_easys[i] for i in sampleNO]
                #     each_feature_easys.clear()
                #     each_feature_easys = aftersample
                #     self.decay = self.decay*0.98
                self.features_easys[feature['feature_id']] = copy(each_feature_easys)  # 所有因子
        if binary_featureid!=None:  # 对于所有参数化双因子，只要tau_and_regression_bound >0 的证据，添加self.features_easys ，基本为（0，-10）
            evidence1_count = 0
            for each_evi in self.features_easys[binary_featureid]:  # 滑动窗口的大小是证据个数
                if each_evi[1] > 0:
                    evidence1_count+=1
            for index in range(int(evidence1_count)):
                self.features_easys[binary_featureid].append((0.01,-10))
            print('加入双因子虚拟证据')

    def influence_modeling(self,update_feature_set):
        '''
        Perform linear regression on the updated feature
        @param update_feature_set:
        @return:
        '''
        # update_feature_set.add(2)
        if len(update_feature_set) > 0:
            self.init_tau_and_alpha(update_feature_set)
            for feature_id in update_feature_set:
                # For some features whose features_easys is empty, regression is none after regression
                if self.features[feature_id]['parameterize'] == 1:
                    if self.features[feature_id]['feature_type'] == 'unary_feature'and self.features[feature_id]['feature_name']!='sliding_window_factor':
                        print('单因子回归')
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id],n_job=self.n_job,factor_type='unary')
                    elif self.features[feature_id]['feature_type'] == 'binary_feature':
                        print('双因子回归')
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=self.n_job,factor_type='binary')
                    else:
                        print('sliding_windows因子回归')
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id],n_job=self.n_job, factor_type='sliding_window')

    def init_tau_and_alpha(self, feature_set):
        '''
        Calculate tau and alpha for a given feature
        @param feature_set:
        @return:
        '''
        if type(feature_set) != list and type(feature_set) != set:
            raise ValueError('feature_set must be set or list')
        else:
            for feature_id in feature_set:
                if self.features[feature_id]['parameterize'] == 1 and self.features[feature_id]['feature_type'] != 'binary_feature':
                    #tau value is fixed as the upper bound
                    self.features[feature_id]["tau"] = self.tau_and_regression_bound
                    weight = self.features[feature_id]["weight"]
                    labelvalue0 = 0
                    num0 = 0
                    labelvalue1 = 0
                    num1 = 0
                    for key in weight:
                        if self.variables[key]["is_evidence"] and self.variables[key]["label"] == 0:
                            labelvalue0 += weight[key][1]
                            num0 += 1
                        elif self.variables[key]["is_evidence"] and self.variables[key]["label"] == 1:
                            labelvalue1 += weight[key][1]
                            num1 += 1
                    if num0 == 0 and num1 == 0:
                        continue
                    if num0 == 0:
                        # If there is no label0 connected to the feature, the value is assigned to the lower bound of value, which is currently set to 1
                        #labelvalue0 = min(set(self.features[feature_id].keys()))
                        tmp = []
                        for k,v in self.features[feature_id]['weight'].items():
                            tmp.append(v[1])
                        nptmp = np.array(tmp)
                        labelvalue0 = nptmp.min()
                    else:
                        # The average value of the feature value with a label of 0
                        labelvalue0 /= num0
                    if num1 == 0:
                        tmp = []
                        #labelvalue1 = max(set(self.features[feature_id].keys()))
                        for k,v in self.features[feature_id]['weight'].items():
                            tmp.append(v[1])
                            nptmp = np.array(tmp)
                            labelvalue1 = nptmp.max()
                    else:
                        # The average value of the feature value with label of 1
                        labelvalue1 /= num1
                    alpha = (labelvalue0 + labelvalue1) / 2
                    # alpha == labelvalue0 == labelvalue1, init influence always equals 0.
                    if labelvalue0 == labelvalue1:
                        if num0 > num1:
                            alpha += 1
                        elif num0 < num1:
                            alpha -= 1
                    self.features[feature_id]["alpha"] = alpha
                else:
                    self.features[feature_id]["tau"] = self.tau_and_regression_bound
                    self.features[feature_id]["alpha"] = 0.5

    def evidential_support_by_regression(self,variable_set,update_feature_set,varname2isevidence,vname2classname_fv,fv2probability):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''

        self.varname2isevidence = varname2isevidence
        self.vname2classname_fv = vname2classname_fv
        self.fv2probability = fv2probability
        self.padding_slidingWindow_factor()
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.separate_feature_value()
        if update_feature_set == None or (type(update_feature_set) ==set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid,feature in enumerate(self.features):
                update_feature_set.add(fid)
        self.influence_modeling(update_feature_set)
        coo_data = self.create_csr_matrix().tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []
        # 构建一整个矩阵
        feature_is_sliding_window = []
        unary_feature_count =0
        for feature in self.features:
            # 首先针对单因子
            if feature['feature_type'] != 'binary_feature':
                unary_feature_count+=1
                if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >0:
                    print('因子{0}回归斜率为{1},截距为{2}:'.format(feature['feature_name'],feature['regression'].regression.coef_[0][0],feature['regression'].regression.intercept_[0]))
                    feature['monotonicity'] = True
                else:
                    feature['monotonicity'] = False
                    print('单因子', feature['feature_name'], '单调性被过滤')
                if feature['regression'].regression is not None and feature['monotonicity'] == True:
                    coefs.append(feature['regression'].regression.coef_[0][0])
                    intercept.append(feature['regression'].regression.intercept_[0])
                    zero_confidence.append(1)
                else:
                    coefs.append(0)
                    intercept.append(0)
                    zero_confidence.append(0)

                Ns.append(feature['regression'].N if feature['regression'].N > feature['regression'].effective_training_count else np.NaN)
                residuals.append(feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
                meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
                if feature['regression'].variance is not None:
                    if  feature['regression'].variance == 0:
                        variance.append(1e-6)
                    else:
                        variance.append(feature['regression'].variance)
                else:
                    variance.append(np.NaN)
        coefs = np.array(coefs)[col]
        intercept = np.array(intercept)[col]
        zero_confidence = np.array(zero_confidence)[col]
        residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                         np.array(variance)[col]
        tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
        confidence = np.ones_like(data)
        confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
        # is_slidingwindow_fv = np.array([False, True])[col]
        # is_rankfactor_fv = np.array([True, False])[col]
        # is_equal_zero = np.where(data<0.0001,True,False)
        # confidence[is_rankfactor_fv] += 0.35
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # 正则化

        # 将计算出的evidential support写回
        csr_evidential_support = csr_matrix((evidential_support, (row, col)), shape=(len(self.variables), unary_feature_count))
        for index, var in enumerate(self.variables):
            for feature_id in var['feature_set']:
                if self.features[feature_id]['feature_type'] =='unary_feature':
                    var['feature_set'][feature_id][0] = csr_evidential_support[index, feature_id]
        # 写回theta
        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        espredict = espredict * zero_confidence
        # 针对sliding_
        is_basic_fv= np.array([True,False])[col]
        if False:
            is_slidingwindow_fv = np.array([False,True])[col]
            is_smaller_0 = np.where(espredict<0,True,False)
            is_sliding_smallerthan_zero = is_slidingwindow_fv & is_smaller_0
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0

        loges = np.log(evidential_support)  # 取自然对数
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), unary_feature_count))

        logunes = np.log(1 - evidential_support)
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), unary_feature_count))

        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        if (len(self.observed_variables_set)/len(self.variables))>0.95:
            espredict[is_basic_fv] = 0
            print('已标注比例超过99% 开始关掉basic 单因子')
        approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables),unary_feature_count))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)
        # approximate_weight = approximate_weight.toarray()
        # 至此所有单因子 feature 的 approximate_weight 计算完毕
        # if len(self.observed_variables_set) / len(self.variables) > 0:
        if True:
            binary_approximate_weight = np.zeros(shape=(len(self.variables),1), dtype=np.float)
            for feature in self.features:
                if feature['feature_type'] == 'binary_feature':
                    feature['binary_weight'] = {}
                    if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >= 0:
                        print('因子{0}回归斜率为{1},截距为{2}:'.format(feature['feature_name'],
                                                             feature['regression'].regression.coef_[0][0],
                                                             feature['regression'].regression.intercept_[0]))
                        feature['monotonicity'] = True
                    else:
                        feature['monotonicity'] = False
                        print('双因子被杀了')
                    if feature['regression'].regression is not None and feature['monotonicity'] == True:
                        zero_confidence = 1
                    else:
                        zero_confidence = 0
                    Ns = feature['regression'].N if feature['regression'].N > feature['regression'].effective_training_count else np.NaN
                    residuals = feature['regression'].residual if feature['regression'].residual is not None else np.NaN
                    meanX = feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN
                    variance = feature['regression'].variance if feature['regression'].variance is not None else np.NaN
                    regression = feature['regression'].regression
                    if regression != None and zero_confidence == 1 and np.isnan(Ns) == False and np.isnan(residuals) == False and np.isnan(meanX) == False and np.isnan(variance) == False:
                        regression_k = regression.coef_[0][0]
                        regression_b = regression.intercept_[0]
                        for weight in feature['weight']:
                            vid1 = weight[0]
                            vid2 = weight[1]
                            featurevalue = feature['weight'][weight][1]
                            tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(featurevalue - meanX, 2) / variance))
                            confidence = 1 - t.sf(tvalue, (Ns - 2)) * 2
                            evidential_support = (1 + confidence) / 2
                            # 写会双因子 theta
                            self.variables[vid1]['feature_set'][feature['feature_id']][0] = evidential_support
                            self.variables[vid2]['feature_set'][feature['feature_id']][0] = evidential_support
                            thisweight = evidential_support * (regression_k * featurevalue + regression_b) * zero_confidence
                            # thisweight = -1
                            # edit 0406 二分类逻辑
                            if self.variables[vid2]['is_evidence'] == True and thisweight>=0:
                                if vid1 not in feature['binary_weight']:
                                    feature['binary_weight'][vid1] = {}
                                feature['binary_weight'][vid1][weight] = [vid1, vid2, self.variables[vid2]['label'], featurevalue, thisweight]
                                if self.variables[vid2]['label'] == 1:
                                    binary_approximate_weight[vid1][0] += thisweight
                                    # binary_approximate_weight[vid1][1] += 1
                                if self.variables[vid2]['label'] == 0:
                                    binary_approximate_weight[vid1][0] -= thisweight
                                    # binary_approximate_weight[vid1][1] += 1
                                p_es[vid1, 0] *= evidential_support
                                p_unes[vid1, 0] *= (1 - evidential_support)
                            if self.variables[vid1]['is_evidence'] == True and thisweight>=0:
                                if 'binary_weight' not in feature:
                                    feature['binary_weight'] = {}
                                if vid2 not in feature['binary_weight']:
                                    feature['binary_weight'][vid2] = {}
                                feature['binary_weight'][vid2][weight] = [vid2, vid1, self.variables[vid1]['label'], featurevalue, thisweight]
                                if self.variables[vid1]['label'] == 1:
                                    binary_approximate_weight[vid2][0] += thisweight
                                    # binary_approximate_weight[vid2][1] += 1
                                if self.variables[vid1]['label'] == 0:
                                    binary_approximate_weight[vid2][0] -= thisweight
                                    # binary_approximate_weight[vid2][1] += 1
                                p_es[vid2, 0] *= evidential_support
                                p_unes[vid2, 0] *= (1 - evidential_support)
            # if True:
            #     for eachdim in range(0, len(self.variables)):
            #         if binary_approximate_weight[eachdim][1] > 0:
            #             binary_approximate_weight[eachdim][0] /= 1.5
                            #binary_approximate_weight[eachdim][1]
            # binary_approximate_weight = binary_approximate_weight[:, 0].reshape(-1, 1)
            binary_approximate_weight = np.array(binary_approximate_weight.sum(axis=1)).reshape(-1)
            approximate_weight = approximate_weight + binary_approximate_weight
        # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            if 'approximate_probability' not in var.keys():
                var['approximate_probability'] = [.0]
            if 'approximate_weight' not in var.keys():
                var['approximate_weight'] = [.0]
            # if 'approximate_probability_binary' not in var.keys():
            #     var['approximate_probability_binary'] = [.0]
            # if 'approximate_weight_binary' not in var.keys():
            #     var['approximate_weight_binary'] = [.0]
            var['approximate_weight'] = approximate_weight[index]
            # var['approximate_probability_binary'] = binary_approximate_weight[index][1]
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        # 二分类和多分类应该是一致的
        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))
        logging.info("evidential_support_by_regression finished")
    
    def create_csr_matrix(self):
        '''
        创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
        :return:
        '''
        data = list()
        row = list()
        col = list()
        unary_feature_count = 2
        for index, var in enumerate(self.variables):
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                if self.features[feature_id]['feature_type'] != 'binary_feature':
                    data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)  #feature_value
                    row.append(index)
                    col.append(feature_id)

                    # unary_feature_count+=1
        '''
        binary_datas = [None] * len(self.features)
        for feature_id in feature_set:
            if self.features['feature_id']['feature_type'] == 'binary_feature':
                data = list()
                row = list()
                col = list()
                for eachbinary in self.features[feature_id]['weight']:
                    var0 = eachbinary[0]
                    var1 = eachbinary[1]
                    featurevalue = self.features[feature_id]['weight'][eachbinary][1]
                    data.append(featurevalue + self.NOT_NONE_VALUE)
                    row.append(var0)
                    col.append(var1)
                    data.append(featurevalue + self.NOT_NONE_VALUE)
                    row.append(var1)
                    col.append(var0)
            binary_datas[feature_id] = csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.variables)))
        '''
        return csr_matrix((data, (row, col)), shape=(len(self.variables), unary_feature_count))
        # return csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.features))

    def padding_slidingWindow_factor(self):
        print('in evidential_support :padding_slidingWindow_factor')
        factor_id =1
        # for eachfea in self.features:
            # if eachfea['feature_name'] == 'sliding_window_factor':
            #     factor_id =eachfea ['feature_id']
        for vname in self.vname2classname_fv.keys(): #self.vname2classname_fv 放着 fv数值
            classnameset = set(self.vname2classname_fv[vname].keys()) # 感觉是过滤作用
            pairidset = self.vname2pairid[vname]
            for pairid in pairidset:
                thispairclassname = self.variables[pairid]['classname']
                if thispairclassname in classnameset:
                    fv = round(self.vname2classname_fv[vname][thispairclassname],2) # 取fv值
                    self.variables[pairid]['feature_set'][factor_id] = [0.5, fv] # 改写variables里面的fv
                    if pairid not in self.features[factor_id]['weight']:
                        self.features[factor_id]['weight'][pairid] = {}
                    self.features[factor_id]['weight'][pairid] = [0.5, fv] # 改写features里面的fv
                else:
                    dummy_fv = 0
                    self.variables[pairid]['feature_set'][factor_id] = [0.5, dummy_fv] # 改写variables里面的fv
                    if pairid not in self.features[factor_id]['weight']:
                        self.features[factor_id]['weight'][pairid] = {}
                    self.features[factor_id]['weight'][pairid] = [0.5, dummy_fv] # 改写features里面的fv
    def prior_probability(self):
        for eachvar in self.varname2isevidence:
            if self.varname2isevidence[eachvar]:
                thisvarclass = self.varname2classname[eachvar]
            # 领域内有相同标签的字段
                if thisvarclass in self.vaname2classname_fv[eachvar]:
                    thisfv = self.vaname2classname_fv[eachvar][thisvarclass]
                else:
                    thisfv = 0
                # for eachclass in self.vaname2classname_fv[eachvar].keys():
                #     thisfv = self.vaname2classname_fv[eachvar][eachclass]
                if thisvarclass not in self.evidenceclass2fv:
                    self.evidenceclass2fv[thisvarclass] = []
                self.evidenceclass2fv[thisvarclass].append(round(thisfv,2))
        # 开始计算先验概率
        all_right_fv = []
        for eachclass in self.evidenceclass2fv.keys():
            all_right_fv+=self.evidenceclass2fv[eachclass]
        fv2count = {}
        for eachfv in all_right_fv:
            if eachfv not in fv2count.keys():
                fv2count[eachfv] = 1
            else :
                fv2count[eachfv] += 1
        self.fv2probability = {}
        for eachfv in fv2count.keys():
            self.fv2probability[eachfv] = fv2count[eachfv]/len(all_right_fv)
        print('debug')