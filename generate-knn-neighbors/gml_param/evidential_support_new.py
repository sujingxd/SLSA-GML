from copy import copy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import gml_utils
from pyds import MassFunction
import math


# class Regression:
#     '''
#     Calculate evidence support by linear regression
#     '''
#     def __init__(self, evidences, n_job, proportion, effective_training_count_threshold = 2,para=None,factor_type='unary'):
#         '''
#         @param evidences:
#         @param n_job:
#         @param effective_training_count_threshold:
#         '''
#         self.para = para
#         self.effective_training_count = max(2, effective_training_count_threshold)
#         self.n_job = n_job
#         self.proportion = proportion
#         if len(evidences) > 0:
#             XY = np.array(evidences)
#             self.X = XY[:, 0].reshape(-1, 1)
#             self.Y = XY[:, 1].reshape(-1, 1)
#         else:
#             self.X = np.array([]).reshape(-1, 1)
#             self.Y = np.array([]).reshape(-1, 1)
#         self.balance_weight_y0_count = 0
#         self.balance_weight_y1_count = 0
#         for y in self.Y:
#             if y > 0:
#                 self.balance_weight_y1_count += 1
#             else:
#                 self.balance_weight_y0_count += 1
#         self.perform()
#
#     def perform(self):
#         '''
#         Perform linear regression
#         @return:
#         '''
#         self.N = np.size(self.X)
#         if self.N <= self.effective_training_count: # not enter
#             self.regression = None
#             self.residual = None
#             self.meanX = None
#             self.variance = None
#             self.k = None
#             self.b = None
#         else:
#             sample_weight_list = None
#             if len(np.unique(self.X)) == 1: # not enter
#                 # single-value factor balanced
#                 sample_weight_list = list()
#                 for y in self.Y:
#                     if y[0] > 0:
#                         sample_weight_list.append(self.proportion)
#                     else:
#                         sample_weight_list.append(1)
#                 self.Y = np.array(self.Y, dtype=np.float)
#                 # if type(self.para['rule_damping_coef']) != list:
#                 #     self.Y *= self.para['rule_damping_coef']
#                 # else:
#                 #     if self.balance_weight_y1_count < self.balance_weight_y0_count:
#                 #         self.Y *= self.para['rule_damping_coef'][0]
#                 #     else:
#                 #         self.Y *= self.para['rule_damping_coef'][1]
#             else:
#                 if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0: # enter
#                     sample_weight_list = list()
#                     sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count*40
#                     for y in self.Y:
#                         if y[0] > 0:
#                             sample_weight_list.append(sample_weight)
#                         else:
#                             sample_weight_list.append(1)
#                 self.Y = np.array(self.Y, dtype=np.float)
#             self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y, sample_weight=sample_weight_list)
#
#             self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
#             self.meanX = np.mean(self.X)  #The average value of feature_value of all evidence variables of this feature
#             self.variance = np.sum((self.X - self.meanX) ** 2)
#             self.k = self.regression.coef_[0][0]
#             self.b = self.regression.intercept_[0]

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
            #     # blocking = 10
            #     # sample_weight = blocking
                for y in self.Y:
                    if y[0] > 0:
                        self.sample_weight_list.append(sample_weight)
                    else:
                        self.sample_weight_list.append(1)
                # print('sliding_window Weight', sample_weight)
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
    def __init__(self,variables,features,para):
        '''
        @param variables:
        @param features:
        '''
        self.para = para
        self.variables = variables
        import copy
        self.ori_variables = copy.deepcopy(variables)
        self.ori_features = copy.deepcopy(features)
        self.features = features   #
        self.features_easys = dict()  # Store all easy feature values of all features    :feature_id:[[value1,bound],[value2,bound]...]
        self.tau_and_regression_bound = 10
        self.NOT_NONE_VALUE = 1e-8
        self.n_job = 10
        self.delta = 2
        self.effective_training_count_threshold = 2
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
        self.observed_variables_set = set()
        self.poential_variables_set = set()
        self.train_ids = gml_utils.get_train_ids(variables)
        self.padding_slidingWindow_factor(self.train_ids, 1)  ## sj
        self.cnt = 0

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
        for rel,edges in dict_reltype_edges.items():
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

 #    def evidential_support(self,variable_set,update_feature_set):
 #        '''
 #        Evidence support for fusing all factors of each latent variable ,Only update the changed features each time
 #        @param variable_set:
 #        @param update_feature_set:
 #        @return:
 #        '''
 #        # Select different mass functions according to factor types and calculate evidence support
 # #       self.dict_rel_acc = self.get_dict_rel_acc()
 # #       self.get_unlabeled_var_feature_evi()
 #        mass_functions = list()
 #        self.computer_unary_feature_es(update_feature_set)
 #        for vid in variable_set:
 #            var = self.variables[vid]
 #            for fid in self.variables[vid]['feature_set']:
 #                if self.features[fid]['feature_type'] == 'unary_feature': #Judgment factor type
 #                    if self.features[fid]['parameterize']  == 1 : # Whether the factor is parameterized
 #                        mass_functions.append(self.construct_mass_function_for_para_feature(var['feature_set'][fid][0]))
 #                    else :
 #                        if 'unary_feature_evi_prob' in self.variables[vid]:
 #                            if len(var['unary_feature_evi_prob']) > 0:
 #                                for (feature_id, feature_name, n_samples, neg_prob, pos_prob) in var['unary_feature_evi_prob']:
 #                                    mass_functions.append(self.construct_mass_function_for_propensity(self.word_evi_uncer_degree, max(pos_prob, neg_prob),min(pos_prob, neg_prob)))
 #                if self.features[fid]['feature_type'] == 'binary_feature':
 #                    if 'binary_feature_evi' in var:
 #                        if len(var['binary_feature_evi']) > 0:
 #                            for (anotherid, feature_name, feature_id) in var['binary_feature_evi']:
 #                                rel_acc = self.dict_rel_acc[feature_name]
 #                                mass_functions.append(self.construct_mass_function_for_propensity(self.relation_evi_uncer_degree,rel_acc, 1-rel_acc))
 #            if len(mass_functions) > 0: #Calculate evidence support by D-S theory
 #                combine_evidential_support = self.labeling_propensity_with_ds(mass_functions)
 #                var['evidential_support'] = combine_evidential_support['l']
 #                mass_functions.clear()
 #            else:
 #                var['evidential_support'] = 0.0
 #        logging.info("evidential_support calculate finished")

    # def separate_feature_value(self):
    #     '''
    #     Select the easy feature value of each feature for linear regression
    #     :return:
    #     '''
    #     evidences = list()
    #     self.features_easys.clear()
    #     for feature in self.features:
    #         if feature['parameterize'] == 1:
    #             evidences.clear()
    #             if feature['feature_type'] == 'unary_feature':
    #                 for var_id, value in feature['weight'].items():
    #                     if var_id in self.observed_variables_set:
    #                         evidences.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
    #             else:
    #                 ## ???????????
    #                 for var_ids, value in feature['weight'].items():
    #                     if var_ids[0] in self.observed_variables_set and var_ids[1] in self.observed_variables_set:
    #                         evidences.append([value[1], (1 if self.variables[var_ids[0]]['label'] == self.variables[var_ids[1]]['label'] else -1) * self.tau_and_regression_bound])
    #                         # if self.variables[var_ids[0]]['label'] != self.variables[var_ids[1]]['label']:
    #                             # print("***********")
    #             self.features_easys[feature['feature_id']] = copy(evidences)
    #             # evidences_cp = np.array(evidences)
    #             # np.savetxt('evidences{}.txt'.format(self.cnt), evidences_cp, fmt="%s", delimiter="\t")
    #             self.cnt += 1

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
        observed_variables_list = list(self.observed_variables_set)
        observed_variables_list.sort()
        for feature in self.features:
            if feature['feature_name'] =='sliding_window_factor': # 滑动窗口easy直接放到sliding_window_feature_easys
                sliding_window_feature_easys=[]
                for var_id,value in feature['weight'].items():
                    if type(var_id) == int:   # 单因子
                        if var_id in observed_variables_list:
                            # sliding_window_feature_easys.append([value[1],((self.fv2probability[value[1]]-1) if abs(value[1]-0.0)<0.01 else self.fv2probability[value[1]])*20])
                            sliding_window_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
                            # sliding_window_feature_easys.append([value[1], self.variables[var_id]['label']*self.tau_and_regression_bound])
                    # if type(var_id) ==tuple: # 双因子
                    #     binary_featureid = feature['feature_id']
                    #     if var_id[0] in self.observed_variables_set and var_id[1] in self.observed_variables_set:
                    #         if self.variables[var_id[0]]['label'] == self.variables[var_id[1]]['label']:
                    #             sliding_window_feature_easys.append([value[1], 1*self.tau_and_regression_bound])
                    #         else :
                    #             sliding_window_feature_easys.append([value[1], -1*self.tau_and_regression_bound])

                print('sliding_window evidence: ',len(sliding_window_feature_easys))
                # self.features_easys[1] = copy(sliding_window_feature_easys)  # 这里强制 feature_id = 1
                self.features_easys[1] = copy(sliding_window_feature_easys)  # 这里强制 feature_id = 1
            if feature['parameterize'] == 1 and feature['feature_name']!='sliding_window_factor':
                each_feature_easys.clear()  # 不是滑动窗口easy的直接放到each_feature_easys
                for var_id, value in feature['weight'].items():
                    if type(var_id) ==int: # 单因子
                        if var_id in self.observed_variables_set:
                            each_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
                            # each_feature_easys.append([value[1], self.variables[var_id]['label']* self.tau_and_regression_bound])
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

    # def influence_modeling(self,update_feature_set):
    #     '''
    #     Perform linear regression on the updated feature
    #     @param update_feature_set:
    #     @return:
    #     '''
    #     if len(update_feature_set) > 0:
    #         self.init_tau_and_alpha(update_feature_set)
    #         for feature_id in update_feature_set:
    #             # For some features whose features_easys is empty, regression is none after regression
    #             if self.features[feature_id]['parameterize'] == 1:
    #                 if self.features[feature_id]['feature_type'] != 'binary_feature':
    #                     self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id],n_job=self.n_job,proportion=self.proportion,para =self.para)
    #                 else:
    #                     self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=self.n_job, proportion=None,para =self.para)
    #                     # import matplotlib.pyplot as plt
    #                     # plt.scatter(self.features[feature_id]['regression'].X, (self.features[feature_id]['regression'].k) * (self.features[feature_id]['regression'].X)+ (self.features[feature_id]['regression'].b))
    #                     # plt.show()

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
                    if self.features[feature_id]['feature_type'] == 'unary_feature'and self.features[feature_id]['feature_name']!='sliding_window_factor': # 他们的是滑动单因子
                        print('单因子回归')
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id],n_job=self.n_job,factor_type='unary')
                    elif self.features[feature_id]['feature_type'] == 'binary_feature' and self.features[feature_id]['feature_name']!='sliding_window_factor':
                        print('双因子回归')
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=self.n_job,factor_type='binary')
                    else:
                        print('sliding_windows因子回归')
                        self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id],n_job=self.n_job, factor_type='sliding_window')


    def padding_slidingWindow_factor(self,train_ids, windows_feature_id):
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        # for var_id in list(self.observed_variables_set)+list(self.poential_variables_set):
        for var_id in list(self.observed_variables_set)+list(self.poential_variables_set):
            train_neighbors_cnt = 0
            test_neighbors_cnt = 0
            neighbor_pos = 0
            neighbor_neg = 0
            for each_feature_id in self.ori_variables[var_id]['feature_set'].keys():
                if each_feature_id != windows_feature_id:
                    continue
                neighbors_pair_list = self.ori_variables[var_id]['feature_set'][each_feature_id]
                for each_neighbors_pair in neighbors_pair_list:
                    left_id = each_neighbors_pair[0]
                    right_id = each_neighbors_pair[1]
                    if left_id == var_id:
                        other_id = right_id
                    else:
                        other_id = left_id
                    if other_id in train_ids:
                        train_neighbors_cnt += 1
                    else:
                        test_neighbors_cnt += 1
                    if self.variables[other_id]['label'] == 0 and self.variables[other_id]['label'] != -1:
                        neighbor_neg += 1
                    elif self.variables[other_id]['label'] == 1 and self.variables[other_id]['label'] != -1:
                        neighbor_pos += 1
                    else:
                        pass
                    if train_neighbors_cnt >4:
                        break
            if each_feature_id != 0:
                if neighbor_pos + neighbor_neg == 0:
                    fv = 0.5
                    self.features[1]['weight'][var_id] = [0.5, fv]  # sj  modify
                    self.variables[var_id]['feature_set'][each_feature_id] = [0.5, fv]
                else:
                    fv = neighbor_pos/(neighbor_pos + neighbor_neg)
                    self.features[1]['weight'][var_id] = [0.5 if fv > 0.5 else -0.5, fv]  # sj  modify
                    self.variables[var_id]['feature_set'][each_feature_id] = [0.5 if fv >0.5 else -0.5, fv]
        import copy
        tmp = copy.deepcopy(self.features)
        for each_pair in self.features[each_feature_id]['weight']:
            if type(each_pair) == tuple:
                del tmp[each_feature_id]['weight'][each_pair]
        self.features = tmp
        tmp_var = self.variables
            # else:
            #     dummy_fv = 0
            #     self.features[each_feature_id]['weight'][var_id] = [0.5, dummy_fv]
            #     self.variables[var_id]['feature_set'][each_feature_id] = [0.5, dummy_fv]

    # def init_tau_and_alpha(self, feature_set):
    #     '''
    #     Calculate tau and alpha for a given feature
    #     @param feature_set:
    #     @return:
    #     '''
    #     if type(feature_set) != list and type(feature_set) != set:
    #         raise ValueError('feature_set must be set or list')
    #     else:
    #         for feature_id in feature_set:
    #             if self.features[feature_id]['parameterize'] == 1 and self.features[feature_id]['feature_type'] != 'binary_feature': # not enter
    #             # if self.features[feature_id]['parameterize'] == 1:  # sujing
    #                 #tau value is fixed as the upper bound
    #                 self.features[feature_id]["tau"] = self.tau_and_regression_bound
    #                 weight = self.features[feature_id]["weight"]
    #                 labelvalue0 = 0
    #                 num0 = 0
    #                 labelvalue1 = 0
    #                 num1 = 0
    #                 for key in weight:
    #                     if self.variables[key]["is_evidence"] and self.variables[key]["label"] == 0:
    #                         labelvalue0 += weight[key][1]
    #                         num0 += 1
    #                     elif self.variables[key]["is_evidence"] and self.variables[key]["label"] == 1:
    #                         labelvalue1 += weight[key][1]
    #                         num1 += 1
    #                 if num0 == 0 and num1 == 0:
    #                     continue
    #                 if num0 == 0:
    #                     # If there is no label0 connected to the feature, the value is assigned to the lower bound of value, which is currently set to 1
    #                     #labelvalue0 = min(set(self.features[feature_id].keys()))
    #                     tmp = []
    #                     for k,v in self.features[feature_id]['weight'].items():
    #                         tmp.append(v[1])
    #                     nptmp = np.array(tmp)
    #                     labelvalue0 = nptmp.min()
    #                 else:
    #                     # The average value of the feature value with a label of 0
    #                     labelvalue0 /= num0
    #                 if num1 == 0:
    #                     tmp = []
    #                     #labelvalue1 = max(set(self.features[feature_id].keys()))
    #                     for k,v in self.features[feature_id]['weight'].items():
    #                         tmp.append(v[1])
    #                         nptmp = np.array(tmp)
    #                         labelvalue1 = nptmp.max()
    #                 else:
    #                     # The average value of the feature value with label of 1
    #                     labelvalue1 /= num1
    #                 alpha = (labelvalue0 + labelvalue1) / 2
    #                 # alpha == labelvalue0 == labelvalue1, init influence always equals 0.
    #                 if labelvalue0 == labelvalue1:
    #                     if num0 > num1:
    #                         alpha += 1
    #                     elif num0 < num1:
    #                         alpha -= 1
    #                 self.features[feature_id]["alpha"] = alpha
    #             else:
    #                 self.features[feature_id]["tau"] = self.tau_and_regression_bound  # enter  10
    #                 self.features[feature_id]["alpha"] = 0.5




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
                if self.features[feature_id]['parameterize'] == 1 and self.features[feature_id]['feature_type'] != 'binary_feature' and self.features[feature_id]['feature_type'] != 'unary_feature': # not enter
                # if self.features[feature_id]['parameterize'] == 1:  # sujing
                    #tau value is fixed as the upper bound
                    self.features[feature_id]["tau"] = self.tau_and_regression_bound
                    weight = self.ori_features[feature_id]["weight"]
                    labelvalue0 = 0
                    num0 = 0
                    labelvalue1 = 0
                    num1 = 0
                    for key in weight:
                        if self.variables[key[0]]["is_evidence"] and self.variables[key[1]]["is_evidence"] and self.variables[key[0]]["label"]!= self.variables[key[1]]["label"] :
                            labelvalue0 += weight[key][1]
                            num0 += 1
                        elif self.variables[key[0]]["is_evidence"] and self.variables[key[1]]["is_evidence"] and self.variables[key[0]]["label"]== self.variables[key[1]]["label"]:
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
                    self.features[feature_id]["tau"] = self.tau_and_regression_bound  # enter  10
                    self.features[feature_id]["alpha"] = 0.5



    def evidential_support_by_regression(self,variable_set, update_feature_set):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        num0 = 0
        num1 = 0
        c_tl_1 = 0
        c_l_1 = 0
        c_tll_1 = 0
        tl_1 = 0
        for i in range(0, len(self.variables)):
            var = self.variables[i]
            if var['is_evidence']:
                if var['label'] == 1:
                    num1 += 1
                    if var['is_easy'] == False or var['is_hard_easy'] == True:
                        c_l_1 += 1
                        if var['true_label'] == 1:
                            c_tll_1 += 1
                else:
                    num0 += 1
                if var['is_easy'] == False or var['is_hard_easy'] == True:
                    if var['true_label'] == 1:
                        c_tl_1 += 1
            if (var['is_easy'] == False or var['is_hard_easy'] == True) and var['true_label'] == 1:
                tl_1 += 1
        if num0 == 0 or num1 == 0:
            self.proportion = 1
        else:
            self.proportion = math.pow(float(num0 / num1), self.para['instance_balance_coef'])
        c_recall = None
        c_precition = None
        c_F1 = None
        if c_l_1 > 0 and c_tl_1 > 0:
            c_recall = round(float(c_tll_1) / c_tl_1, 5)
            c_precition = round(float(c_tll_1) / c_l_1, 5)
            c_F1 = round(float(c_tll_1) / (c_tl_1 + c_l_1) * 2, 5)
            print('left 1: ', tl_1 - c_tl_1)
            print('c: ', c_recall, c_precition, c_F1)

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
        for feature in self.features:
            if feature['feature_type'] != 'binary_feature':
                if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >=0:
                    feature['monotonicity'] = True
                else:
                    feature['monotonicity'] = False
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
                    if feature['regression'].variance == 0:
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
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # 正则化
        # 将计算出的evidential support写回
        csr_evidential_support = csr_matrix((evidential_support, (row, col)), shape=(len(self.variables), len(self.features)))

        for index, var in enumerate(self.variables):
            for feature_id in var['feature_set']:
                var['feature_set'][feature_id][0] = csr_evidential_support[index, feature_id]

        #csr_evidential_support[:,0] = csr_evidential_support[:,0].todense()  *  xg_killer

        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0

        loges = np.log(evidential_support)  # 取自然对数
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), len(self.features)))

        logunes = np.log(1 - evidential_support)
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), len(self.features)))

        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))

        approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))

        # reweighting
        if self.para['reweighting_balance'] != None:
            rule_cols = []
            other_cols = []
            for feature in self.features:
                if feature['feature_type'] != 'binary_feature' and 'aseline' not in feature['feature_name']:
                    rule_cols.append(feature['feature_id'])
                else:
                    other_cols.append(feature['feature_id'])

            espredict_rule = approximate_weight[:, rule_cols]
            espredict_values = espredict_rule.tocoo().data
            damping_1 = np.zeros_like(espredict_values)
            damping_1[np.where(espredict_values > 0)] = 1
            damping_1 = csr_matrix((damping_1, (espredict_rule.tocoo().row, espredict_rule.tocoo().col)), shape=(espredict_rule.shape[0],espredict_rule.shape[1]))
            damping_1 = float(1) / np.array(damping_1.sum(axis=1)) ** (self.para['reweighting_balance'])
            damping_1[np.where(np.isinf(damping_1) == True)] = 0
            damping_0 = np.zeros_like(espredict_values)
            damping_0[np.where(espredict_values < 0)] = 1
            damping_0 = csr_matrix((damping_0, (espredict_rule.tocoo().row, espredict_rule.tocoo().col)), shape=(espredict_rule.shape[0],espredict_rule.shape[1]))
            damping_0 = float(1) / np.array(damping_0.sum(axis=1)) ** (self.para['reweighting_balance'])
            damping_0[np.where(np.isinf(damping_0) == True)] = 0

            approximate_weight = approximate_weight.todense().reshape(len(self.variables), len(self.features))
            array_rule_cols = np.array(rule_cols).reshape(1, -1)
            for eachrowid in range(0, len(self.variables)):
                approximate_weight[eachrowid, array_rule_cols[np.where(approximate_weight[eachrowid, rule_cols] < 0)]] *= damping_0[eachrowid, 0]
                approximate_weight[eachrowid, array_rule_cols[np.where(approximate_weight[eachrowid, rule_cols] > 0)]] *= damping_1[eachrowid, 0]

        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)
        # 至此所有单因子 feature 的 approximate_weight 计算完毕
        ##############################################################################################################
        binary_approximate_weight = np.zeros(shape=(len(self.variables),), dtype=np.float)
        for feature in self.features:
            if feature['feature_type'] == 'binary_feature':
                if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >= 0:
                    feature['monotonicity'] = True
                else:
                    feature['monotonicity'] = False
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
                        thisweight = evidential_support * (regression_k * featurevalue + regression_b) * zero_confidence
                        if self.variables[vid2]['is_evidence'] == True:
                            if thisweight >= 0:
                                # vid1 should equal vid2
                                if self.variables[vid2]['label'] == 0:
                                    # vid2 = -1, vid1 = -1;  vid2 = 1, vid1 = 1.
                                    thisweight = 0 - math.fabs(thisweight)
                            else:
                                if feature['polar'] == 1:
                                    thisweight = 0
                                # vid1 should opposite to vid2
                                if self.variables[vid2]['label'] == 0:
                                    # vid2 = -1, vid1 = 1;  vid2 = 1, vid1 = -1.
                                    thisweight = math.fabs(thisweight)
                            if thisweight >= 0:
                                thisweight *= self.proportion
                            if 'binary_weight' not in feature:
                                feature['binary_weight'] = {}
                            thisweight *= self.para['binary_damping_coef']
                            if vid1 not in feature['binary_weight']:
                                feature['binary_weight'][vid1] = {}
                            feature['binary_weight'][vid1][weight] = [vid1, vid2, self.variables[vid2]['label'], featurevalue, thisweight]
                            binary_approximate_weight[vid1] += thisweight
                            p_es[vid1, 0] *= evidential_support
                            p_unes[vid1, 0] *= (1 - evidential_support)
                        if self.variables[vid1]['is_evidence'] == True:
                            if thisweight >= 0:
                                # vid2 should equal vid1
                                if self.variables[vid1]['label'] == 0:
                                    # vid1 = -1, vid2 = -1;  vid1 = 1, vid2 = 1.
                                    thisweight = 0 - math.fabs(thisweight)
                            else:
                                if feature['polar'] == 1:
                                    thisweight = 0
                                # vid2 should opposite to vid1
                                if self.variables[vid1]['label'] == 0:
                                    # vid1 = -1, vid2 = 1;  vid1 = 1, vid2 = -1.
                                    thisweight = math.fabs(thisweight)
                            if thisweight >= 0:
                                thisweight *= self.proportion
                            if 'binary_weight' not in feature:
                                feature['binary_weight'] = {}
                            thisweight *= self.para['binary_damping_coef']
                            if vid2 not in feature['binary_weight']:
                                feature['binary_weight'][vid2] = {}
                            feature['binary_weight'][vid2][weight] = [vid2, vid1, self.variables[vid1]['label'], featurevalue, thisweight]
                            binary_approximate_weight[vid2] += thisweight
                            p_es[vid2, 0] *= evidential_support
                            p_unes[vid2, 0] *= (1 - evidential_support)
        approximate_weight = approximate_weight + binary_approximate_weight
        # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            var['approximate_weight'] = approximate_weight[index]
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))
        logging.info("evidential_support_by_regression finished")
    
    # def create_csr_matrix(self):
    #     '''
    #     创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
    #     :return:
    #     '''
    #     data = list()
    #     row = list()
    #     col = list()
    #     for index, var in enumerate(self.variables):
    #         feature_set = self.variables[index]['feature_set']
    #         for feature_id in feature_set:
    #             if self.features[feature_id]['feature_type'] != 'binary_feature':
    #                 data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)
    #                 row.append(index)
    #                 col.append(feature_id)
    #
    #     '''
    #     binary_datas = [None] * len(self.features)
    #     for feature_id in feature_set:
    #         if self.features['feature_id']['feature_type'] == 'binary_feature':
    #             data = list()
    #             row = list()
    #             col = list()
    #             for eachbinary in self.features[feature_id]['weight']:
    #                 var0 = eachbinary[0]
    #                 var1 = eachbinary[1]
    #                 featurevalue = self.features[feature_id]['weight'][eachbinary][1]
    #                 data.append(featurevalue + self.NOT_NONE_VALUE)
    #                 row.append(var0)
    #                 col.append(var1)
    #                 data.append(featurevalue + self.NOT_NONE_VALUE)
    #                 row.append(var1)
    #                 col.append(var0)
    #         binary_datas[feature_id] = csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.variables)))
    #     '''
    #     return csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.features)))


    # def create_csr_matrix(self):
    #     '''
    #     创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
    #     :return:
    #     '''
    #     # data = list()
    #     # row = list()
    #     # col = list()
    #     # for index, var in enumerate(self.variables):
    #     #     feature_set = self.variables[index]['feature_set']
    #     #     for feature_id in feature_set:
    #     #         if self.features[feature_id]['feature_type'] != 'binary_feature':
    #     #             data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)
    #     #             row.append(index)
    #     #             col.append(feature_id)
    #     binary_datas = [None] * len(self.features)
    #     for feature_id in [0]:    ## ????????????????
    #         if self.features[feature_id]['feature_type'] == 'binary_feature':
    #             data = list()
    #             row = list()
    #             col = list()
    #             for eachbinary in self.features[feature_id]['weight']:
    #                 var0 = eachbinary[0]
    #                 var1 = eachbinary[1]
    #                 featurevalue = self.features[feature_id]['weight'][eachbinary][1]
    #                 data.append(featurevalue + self.NOT_NONE_VALUE)
    #                 row.append(var0)
    #                 col.append(var1)
    #                 data.append(featurevalue + self.NOT_NONE_VALUE)
    #                 row.append(var1)
    #                 col.append(var0)
    #         binary_datas[feature_id] = csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.variables)))
    #     return binary_datas[0]

    def my_create_csr_matrix(self):
            data = list()
            row = list()
            col = list()
            for index, var in enumerate(self.variables):
                feature_set = self.variables[index]['feature_set']
                for feature_id in feature_set:
                    if self.features[feature_id]['feature_type'] != 'binary_feature':
                        data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)
                        row.append(index)
                        col.append(feature_id)
            return data, row, col


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



    ## wyy
    def evidential_support(self, variable_set, update_feature_set,train_ids):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        ################################################################################################################
        # num0 = 0
        # num1 = 0
        # c_tl_1 = 0
        # c_l_1 = 0
        # c_tll_1 = 0
        # tl_1 = 0
        # for i in range(0, len(self.variables)):
        #     var = self.variables[i]
        #     if var['is_evidence']:
        #         if var['label'] == 1:
        #             num1 += 1
        #             if var['is_easy'] == False or var['is_hard_easy'] == True:
        #                 c_l_1 += 1
        #                 if var['true_label'] == 1:
        #                     c_tll_1 += 1
        #         else:
        #             num0 += 1
        #         if var['is_easy'] == False or var['is_hard_easy'] == True:
        #             if var['true_label'] == 1:
        #                 c_tl_1 += 1
        #     if (var['is_easy'] == False or var['is_hard_easy'] == True) and var['true_label'] == 1:
        #         tl_1 += 1
        # if num0 == 0 or num1 == 0:
        #     self.proportion = 1
        # else:
        #     self.proportion = math.pow(float(num0 / num1), self.para['instance_balance_coef'])

        ## wyy
        self.proportion = 10
        # self.padding_slidingWindow_factor(train_ids)
        ################################################################################################################
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.separate_feature_value()
        if update_feature_set == None or (
                type(update_feature_set) == set() and len(update_feature_set) == 0):  # not enter
            update_feature_set = set()
            for fid, feature in enumerate(self.features):
                update_feature_set.add(fid)
        self.influence_modeling(update_feature_set)  # enter
        # coo_data = self.create_csr_matrix().tocoo()   ## ?????????????????
        # row, col, data = coo_data.row, coo_data.col, coo_data.data

        data, row, col = self.my_create_csr_matrix()
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []
        for feature in self.features:
            if feature['feature_type'] != 'binary_feature':
                if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >= 0:
                    feature['monotonicity'] = True
                else:
                    feature['monotonicity'] = False
                if feature['regression'].regression is not None and feature['monotonicity'] == True:
                    coefs.append(feature['regression'].regression.coef_[0][0])
                    intercept.append(feature['regression'].regression.intercept_[0])
                    zero_confidence.append(1)
                else:
                    coefs.append(0)
                    intercept.append(0)
                    zero_confidence.append(0)
                Ns.append(feature['regression'].N if feature['regression'].N > feature[
                    'regression'].effective_training_count else np.NaN)
                residuals.append(
                    feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
                meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
                if feature['regression'].variance is not None:
                    if feature['regression'].variance == 0:
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
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # 正则化
        # # 将计算出的evidential support写回
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.features)))
        #
        for index, var in enumerate(self.variables):
            for feature_id in var['feature_set']:
                var['feature_set'][feature_id][0] = csr_evidential_support[index, feature_id]

        ##############################################################################################################
        # csr_evidential_support[:,0] = csr_evidential_support[:,0].todense()  *  xg_killer

        # # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence
        # # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0
        #
        loges = np.log(evidential_support)  # 取自然对数
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), len(self.features)))
        #
        logunes = np.log(1 - evidential_support)
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), len(self.features)))

        ## @@@@@@@@@@@
        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)  # 这里计算出的值都比较小

        ## @@@@@@@@@@@
        # p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        # p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        ## @@@@@@@@@@@

        # p_es = np.exp(np.zeros((len(self.variables), 1)))
        # p_unes = np.exp(np.zeros((len(self.variables), 1)))

        ##############################################################################################################
        binary_approximate_weight = np.zeros(shape=(len(self.variables),), dtype=np.float)
        delta = self.delta
        for feature in self.features:
            if feature['feature_type'] == 'binary_feature' and feature['parameterize']==1:
                if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >= 0:
                    feature['monotonicity'] = True
                else:
                    feature['monotonicity'] = False
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
                        confidence = 1 - t.sf(tvalue, (Ns - 2)) * 2  #
                        evidential_support = (1 + confidence) / 2
                        thisweight = evidential_support * (regression_k * featurevalue + regression_b) * zero_confidence
                        if self.variables[vid2]['is_evidence'] == True:
                            if thisweight >= 0:
                                # vid1 should equal vid2
                                if self.variables[vid2]['label'] == 0:
                                    # vid2 = -1, vid1 = -1;  vid2 = 1, vid1 = 1.
                                    thisweight = 0 - math.fabs(thisweight)
                            else:
                                # if feature['polar'] == 1:
                                #     thisweight = 0
                                # vid1 should opposite to vid2
                                if self.variables[vid2]['label'] == 0:
                                    # vid2 = -1, vid1 = 1;  vid2 = 1, vid1 = -1.
                                    thisweight = math.fabs(thisweight)
                            if thisweight >= 0:
                                thisweight *= self.proportion
                            if 'binary_weight' not in feature:
                                feature['binary_weight'] = {}
                            # thisweight *= self.para['binary_damping_coef']
                            if vid1 not in feature['binary_weight']:
                                feature['binary_weight'][vid1] = {}
                            feature['binary_weight'][vid1][weight] = [vid1, vid2, self.variables[vid2]['label'], featurevalue, thisweight]
                            binary_approximate_weight[vid1] += thisweight   ################
                            p_es[vid1, 0] *= evidential_support
                            p_unes[vid1, 0] *= (1 - evidential_support)
                        if self.variables[vid1]['is_evidence'] == True:
                            if thisweight >= 0:
                                # vid2 should equal vid1
                                if self.variables[vid1]['label'] == 0:
                                    # vid1 = -1, vid2 = -1;  vid1 = 1, vid2 = 1.
                                    thisweight = 0 - math.fabs(thisweight)
                            else:
                                # if feature['polar'] == 1:
                                #     thisweight = 0
                                # vid2 should opposite to vid1
                                if self.variables[vid1]['label'] == 0:
                                    # vid1 = -1, vid2 = 1;  vid1 = 1, vid2 = -1.
                                    thisweight = math.fabs(thisweight)
                            if thisweight >= 0:
                                thisweight *= self.proportion
                            if 'binary_weight' not in feature:
                                feature['binary_weight'] = {}
                            # thisweight *= self.para['binary_damping_coef']
                            if vid2 not in feature['binary_weight']:
                                feature['binary_weight'][vid2] = {}
                            feature['binary_weight'][vid2][weight] = [vid2, vid1, self.variables[vid1]['label'], featurevalue, thisweight]
                            binary_approximate_weight[vid2] += thisweight      ################
                            p_es[vid2, 0] *= evidential_support
                            p_unes[vid2, 0] *= (1 - evidential_support)
        ##############################################################################################################

        ## wyy
        # approximate_weight = approximate_weight + binary_approximate_weight
        ## wyy

        # approximate_weight = approximate_weight  # 7132
        # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            var['approximate_weight'] = approximate_weight[index]
            # print(approximate_weight[index])
        ##############################################################################################################
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        ## right
        # for var_id in self.poential_variables_set:
        #     index = var_id
        #     var_p_es = p_es[index]
        #     var_p_unes = p_unes[index]
        #     self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))
        # logging.info("evidential_support_by_regression finished")

        for var_id in self.observed_variables_set.symmetric_difference(self.poential_variables_set):
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            self.variables[index]['evidential_support'] = float(var_p_es / (var_p_es + var_p_unes))
        logging.info("evidential_support_by_regression finished")