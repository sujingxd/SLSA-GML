from copy import copy, deepcopy
from scipy.sparse import *
from scipy.stats import t
from sklearn.linear_model import LinearRegression
import logging
import numpy as np
import gml_utils
from pyds import MassFunction
import time


class Regression_old:
    '''
    线性回归相关类，对所有feature进行线性回归
    todo: feature回归的更新策略:只回归证据支持有变化的feature
    '''
    def __init__(self, each_feature_easys, n_job,effective_training_count_threshold =2):
        '''
        初始化
        @param each_feature_easys:
        @param n_job:
        @param effective_training_count_threshold:
        '''
        self.effective_training_count = max(2, effective_training_count_threshold)
        self.n_job = n_job
        if len(each_feature_easys) > 0:
            XY = np.array(each_feature_easys)
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
        self.perform()

    def perform(self):
        '''
        执行线性回归
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
            sample_weight_list = None
            if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                sample_weight_list = list()
                sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                for y in self.Y:
                    if y[0] > 0:
                        sample_weight_list.append(sample_weight)
                    else:
                        sample_weight_list.append(1)
            self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y,
                                                                                                       sample_weight=sample_weight_list)
            self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
            self.meanX = np.mean(self.X)  # 此feature的所有证据变量的feature_value的平均值
            self.variance = np.sum((self.X - self.meanX) ** 2)
            z = self.regression.predict(np.array([0, 1]).reshape(-1, 1))
            self.k = (z[1] - z[0])[0]
            self.b = z[0][0]

# 规则衰减系数, 建议值： float(1)/平均规则数
rule_damping_coef = float(1)/20
# 分母越大越利于 precision
instance_balance_coef = float(1)
# binary 因子衰减系数，与 k 相关
k = 2
binary_damping_coef = float(1)/(k * 2)
class Regression:
    '''
    Calculate evidence support by linear regression
    '''
    def __init__(self, evidences, n_job, proportion, effective_training_count_threshold = 2):
        '''
        @param evidences:
        @param n_job:
        @param effective_training_count_threshold:
        '''
        self.effective_training_count = max(2, effective_training_count_threshold)
        self.n_job = n_job
        self.proportion = proportion
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
            sample_weight_list = None
            if len(np.unique(self.X)) == 1:
                # single-value factor balanced
                sample_weight_list = list()
                for y in self.Y:
                    if y[0] > 0:
                        sample_weight_list.append(self.proportion)
                    else:
                        sample_weight_list.append(1)
                self.Y = np.array(self.Y, dtype=np.float)
                self.Y *= rule_damping_coef
            elif self.proportion != None:
                if self.balance_weight_y1_count > 0 and self.balance_weight_y0_count > 0:
                    sample_weight_list = list()
                    sample_weight = float(self.balance_weight_y0_count) / self.balance_weight_y1_count
                    for y in self.Y:
                        if y[0] > 0:
                            sample_weight_list.append(sample_weight)
                        else:
                            sample_weight_list.append(1)
            self.regression = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=self.n_job).fit(self.X, self.Y, sample_weight=sample_weight_list)
            self.residual = np.sum((self.regression.predict(self.X) - self.Y) ** 2) / (self.N - 2)
            self.meanX = np.mean(self.X)  #The average value of feature_value of all evidence variables of this feature
            self.variance = np.sum((self.X - self.meanX) ** 2)
            self.k = self.regression.coef_[0][0]
            self.b = self.regression.intercept_[0]

class EvidentialSupport:
    '''
    Evidential Support计算类
    '''
    def __init__(self,variables,features,method = 'regression',evidence_interval_count=10, classNum = -1):
        self.variables = variables
        self.features = features
        self.features_easys = dict()  # 存放所有features的所有easy的featurevalue   :feature_id:[[value1,bound],[value2,bound]...]
        self.tau_and_regression_bound = 10
        self.evidence_interval_count = evidence_interval_count  #区间数为10
        self.NOT_NONE_VALUE = 1e-8
        self.n_job = 10
        self.delta = 2
        self.effective_training_count_threshold = 2
        self.word_evi_uncer_degree = 0.4
        self.relation_evi_uncer_degree = 0.1
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.classNum = classNum
        #根据evidential support方法的不同，分别做不同的初始化
        if method == 'regression':
            self.data_matrix = self.create_csr_matrix()
            self.evidence_interval = gml_utils.init_evidence_interval(self.evidence_interval_count)
            gml_utils.init_evidence(self.features,self.evidence_interval,self.observed_variables_set)
            gml_utils.init_bound(self.variables,self.features)
        if method =='relation':
            self.dict_rel_acc = self.get_dict_rel_acc()


    def get_unlabeled_var_feature_evi(self):
        '''
        计算每个隐变量的每个unary feature相关联的证据变量里面0和1的比例，以及binaryfeature另一端的变量id,
        最终向每个隐变量增加nary_feature_evi_prob和binary_feature_evi两个属性
        :return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        unary_feature_evi_prob = list()
        binary_feature_evi = list()
        for id in self.poential_variables_set:
                for (feature_id,value) in self.variables[id]['feature_set'].items():
                    feature_name = self.features[feature_id]['feature_name']
                    weight = self.features[feature_id]['weight']
                    if self.features[feature_id]['feature_type'] == 'unary_feature':
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
                    elif self.features[feature_id]['feature_type'] == 'binary_feature':
                        for (id1,id2) in weight:
                            anotherid = id1 if id1!=id else id2
                            if self.variables[anotherid]['is_evidence'] == True:
                                binary_feature_evi.append((anotherid,feature_name,feature_id))
                self.variables[id]['unary_feature_evi_prob'] = copy(unary_feature_evi_prob)
                self.variables[id]['binary_feature_evi'] = copy(binary_feature_evi)
                unary_feature_evi_prob.clear()
                binary_feature_evi.clear()

    def separate_feature_value(self):
        '''
        选出每个feature的easy feature value用于线性回归
        :return:
        '''
        each_feature_easys = list()
        self.features_easys.clear()
        for feature in self.features:
            each_feature_easys.clear()
            for var_id, value in feature['weight'].items():
                # 每个feature拥有的easy变量的feature_value
                if var_id in self.observed_variables_set:
                    each_feature_easys.append([value[1], (1 if self.variables[var_id]['label'] == 1 else -1) * self.tau_and_regression_bound])
            self.features_easys[feature['feature_id']] = copy(each_feature_easys)

    def separate_feature_value_single(self, c):
        each_feature_easys = list()
        self.features_easys.clear()
        # 遍历所有因子
        for feature in self.features:
            each_feature_easys.clear()

            fid = feature['feature_id']
            # 找到单因子，确定正样本和负样本
            if feature["feature_type"] == "unary_feature":

                for k, value in feature["weight"].items():

                    if self.variables[k]['is_evidence']:
                        # 文档属于c类
                        if self.variables[k]['label'] == c:
                            each_feature_easys.append([value[1], 1 * self.tau_and_regression_bound])
                        # 文档属于其他类
                        else:
                            each_feature_easys.append([value[1], -1 * self.tau_and_regression_bound])
                self.features_easys[fid] = deepcopy(each_feature_easys)

    def separate_feature_value_binary(self):
        '''
        选出每个feature的easy feature value用于线性回归
        :return:
        '''

        each_feature_easys = list()
        self.features_easys.clear()

        for feature in self.features:
            each_feature_easys.clear()

            fid = feature['feature_id']

            # 找到双因子，确定正样本和负样本
            if feature["feature_type"] == "binary_feature":
                for k, value in feature["weight"].items():
                    # temp = k.split('&')
                    # temp0 = int(temp[0])
                    # temp1 = int(temp[1])
                    temp0 = int(k[0])
                    temp1 = int(k[1])

                    if self.variables[temp0]['is_evidence'] and self.variables[temp1]['is_evidence']:
                        # 两篇文档同类
                        if self.variables[temp0]['label'] == self.variables[temp1]['label']:
                            each_feature_easys.append([value[1], 1 * self.tau_and_regression_bound])
                        else:
                            each_feature_easys.append([value[1], -1 * self.tau_and_regression_bound])
                self.features_easys[fid] = deepcopy(each_feature_easys)

    def create_csr_matrix(self):
        '''
        创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
        :return:
        '''
        data = list()
        row = list()
        col = list()
        for index, var in enumerate(self.variables):
            feature_set = self.variables[index]['feature_set']
            for feature_id in feature_set:
                data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)
                row.append(index)
                col.append(feature_id)
        return csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.features)))

    def create_csr_matrix_single(self, update_feature_set):
        '''
        创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
        :return:
        '''
        data = list()
        row = list()
        col = list()
        col_num = 0
        for fid in update_feature_set:
            if self.features[fid]['feature_name'] == 'distance_feature':
                col_num += 1
                for k, v in self.features[fid]['weight'].items():
                    data.append(v[1] + self.NOT_NONE_VALUE)
                    row.append(k)
                    col.append(fid)
        # for index, var in enumerate(self.variables):
        #     feature_set = self.variables[index]['feature_set']
        #     for feature_id in feature_set:
        #         if self.features[feature_id]['feature_type'] == 'unary_feature':
        #             data.append(feature_set[feature_id][1] + self.NOT_NONE_VALUE)
        #             row.append(index)
        #             col.append(feature_id)
        return csr_matrix((data, (row, col)), shape=(len(self.variables), col_num))

    def create_csr_matrix_binary(self, variable_set, update_feature_set):
        '''
        创建稀疏矩阵存储所有variable的所有featureValue，用于后续计算Evidential Support
        :return:
        '''
        data = list()
        row = list()
        col = list()
        for fid in update_feature_set:
            # print("````````````````````", fid)
            if self.features[fid]['feature_type'] == 'binary_feature':

                for k, v in self.features[fid]['weight'].items():
                    # temp = k.split('&')
                    # first = int(temp[0])
                    # second = int(temp[1])
                    first = int(k[0])
                    second = int(k[1])

                    data.append(v[1] + self.NOT_NONE_VALUE)
                    row.append(first)
                    col.append(second)

                    # data.append(v[1] + self.NOT_NONE_VALUE)
                    # row.append(second)
                    # col.append(first)

        return csr_matrix((data, (row, col)), shape=(len(self.variables), len(self.variables)))

    def influence_modeling(self,update_feature_set):
        '''
        对已更新feature进行线性回归
        @param update_feature_set:
        @return:
        '''
        if len(update_feature_set) > 0:
            self.init_tau_and_alpha(update_feature_set)
            #logging.info("init para finished")
            for feature_id in update_feature_set:
                # 对于某些features_easys为空的feature,回归后regression为none
                self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id], n_job=self.n_job, proportion=None)
            #logging.info("feature regression finished")

    def influence_modeling_multi(self, variable_set,update_feature_set, c=-1):
        '''
        对已更新feature进行线性回归
        @param update_feature_set:
        @return:
        '''
        # for feature in self.features:
        #     if feature['feature_type'] == 'dual':
        #         feature['regression'] = Regression(list(), n_job=self.n_job)

        if len(update_feature_set) > 0:
            self.init_tau_and_alpha_multi(variable_set, update_feature_set, c)
            #logging.info("init para finished")
            for feature_id in update_feature_set:
                # 对于某些features_easys为空的feature,回归后regression为none
                if self.features[feature_id]['feature_type'] == 'binary_feature':
                    self.features[feature_id]['regression'] = Regression(self.features_easys[feature_id],
                                                                         n_job=self.n_job, proportion=None)
                # if self.features[feature_id]['feature_type'] == 'unary_feature':
                #     if 'regression' in self.features[feature_id].keys() :
                #         self.features[feature_id]['regression'][c] = Regression(self.features_easys[feature_id],
                #                                                          n_job=self.n_job)
                #     else:
                #         self.features[feature_id]['regression'] = dict()
                #         self.features[feature_id]['regression'][c] = Regression(self.features_easys[feature_id],
                #                                                                 n_job=self.n_job)

            #logging.info("feature regression finished")

    def init_tau_and_alpha(self, feature_set):
        '''
        对给定的feature计算tau和alpha
        @param feature_set:
        @return:
        '''
        if type(feature_set) != list and type(feature_set) != set:
            print("输入参数错误，应为set或者list")
            return
        else:
            for feature_id in feature_set:
                # tau值固定为上界
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
                    # 如果没有和该feature相连的label0，就把值赋值为value的上界，目前先定为1
                    labelvalue0 = 1
                else:
                    # label为0的featurevalue的平均值
                    labelvalue0 /= num0
                if num1 == 0:
                    # 同上
                    labelvalue1 = 1
                else:
                    # label为1的featurevalue的平均值
                    labelvalue1 /= num1
                alpha = (labelvalue0 + labelvalue1) / 2
                self.features[feature_id]["alpha"] = alpha

    def init_tau_and_alpha_multi(self, variable_set, feature_set, c=-1):
        '''
        对c类给定的feature计算tau和alpha
        @param feature_set:
        @return:
        '''
        if type(feature_set) != list and type(feature_set) != set:
            print("输入参数错误，应为set或者list")
            return
        else:
            for feature_id in feature_set:
                if self.features[feature_id]["feature_type"] == "binary_feature" :
                    # tau值固定为上界
                    self.features[feature_id]["tau"] = self.tau_and_regression_bound

                    weight = self.features[feature_id]["weight"]
                    labelvalue0 = 0
                    num0 = 0
                    labelvalue1 = 0
                    num1 = 0
                    for key in weight.keys():
                        # temp = key.split('&')
                        # first = int(temp[0])
                        # second = int(temp[1])
                        first = int(key[0])
                        second = int(key[1])
                        if self.variables[first]["is_evidence"] and self.variables[second]["is_evidence"]\
                                and self.variables[first]["label"] != self.variables[second]["label"]:
                            labelvalue0 += weight[key][1]
                            num0 += 1
                        elif self.variables[first]["is_evidence"] and self.variables[second]["is_evidence"]\
                                and self.variables[first]["label"] == self.variables[second]["label"]:
                            labelvalue1 += weight[key][1]
                            num1 += 1
                    if num0 == 0 and num1 == 0:
                        self.features[feature_id]["alpha"] = 0
                        continue
                    if num0 == 0:
                        # 如果没有和该feature相连的label0，就把值赋值为value的上界，目前先定为1
                        labelvalue0 = 1
                    else:
                        # label为0的featurevalue的平均值
                        labelvalue0 /= num0
                    if num1 == 0:
                        # 同上
                        labelvalue1 = 1
                    else:
                        # label为1的featurevalue的平均值
                        labelvalue1 /= num1
                    alpha = (labelvalue0 + labelvalue1) / 2
                    self.features[feature_id]["alpha"] = alpha

                if self.features[feature_id]["feature_type"] == "unary_feature":
                    if "tau" not in self.features[feature_id].keys():
                        self.features[feature_id]["tau"] = [0] * self.classNum
                    if "alpha" not in self.features[feature_id].keys():
                        self.features[feature_id]["alpha"] = [0] * self.classNum
                    # tau值固定为上界
                    self.features[feature_id]["tau"][c] = self.tau_and_regression_bound
                    weight = self.features[feature_id]["weight"]
                    labelvalue0 = 0
                    num0 = 0
                    labelvalue1 = 0
                    num1 = 0
                    for key in weight.keys():
                        if self.variables[key]["is_evidence"] and self.variables[key]["label"] != c:
                            labelvalue0 += weight[key][1]
                            num0 += 1
                        elif self.variables[key]["is_evidence"] and self.variables[key]["label"] == c:
                            labelvalue1 += weight[key][1]
                            num1 += 1
                    if num0 == 0 and num1 == 0:
                        self.features[feature_id]["alpha"][c] = 0
                        continue

                    if num0 == 0:
                        # 如果没有和该feature相连的label0，就把值赋值为value的上界，目前先定为1
                        labelvalue0 = 1
                    else:
                        # label为0的featurevalue的平均值
                        labelvalue0 /= num0
                    if num1 == 0:
                        # 同上
                        labelvalue1 = 1
                    else:
                        # label为1的featurevalue的平均值
                        labelvalue1 /= num1
                    alpha = (labelvalue0 + labelvalue1) / 2
                    self.features[feature_id]["alpha"][c] = alpha


    def evidential_support_by_regression(self,variable_set,update_feature_set):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        self.separate_feature_value()
        if update_feature_set == None or (type(update_feature_set) ==set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid,feature in enumerate(self.features):
                update_feature_set.add(fid)
        # 线性回归
        self.influence_modeling(update_feature_set)
        # 准备数据，按照矩阵
        coo_data = self.data_matrix.tocoo()
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
            if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][0] >=0:
                feature['monotonicity'] = True
            else:
                feature['monotonicity'] = False
            if feature['regression'].regression is not None and feature['regression'].variance > 0:
                coefs.append(feature['regression'].regression.coef_[0][0])
                intercept.append(feature['regression'].regression.intercept_[0])
                zero_confidence.append(1)
            else:
                coefs.append(0)
                intercept.append(0)
                zero_confidence.append(0)
            Ns.append(feature['regression'].N if feature['regression'].N > feature[
                'regression'].effective_training_count else np.NaN)
            residuals.append(feature['regression'].residual if feature['regression'].residual is not None else np.NaN)
            meanX.append(feature['regression'].meanX if feature['regression'].meanX is not None else np.NaN)
            variance.append(feature['regression'].variance if feature['regression'].variance is not None else np.NaN)
        coefs = np.array(coefs)[col]
        intercept = np.array(intercept)[col]
        zero_confidence = np.array(zero_confidence)[col]
        residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                         np.array(variance)[col]
        # 公式15，theta f d
        tvalue = float(delta) / (residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance))
        confidence = np.ones_like(data)
        confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # 正则化
        # 将计算出的evidential support写回
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.features)))
        for index, var in enumerate(self.variables):
            for feature_id in var['feature_set']:
                var['feature_set'][feature_id][0] = csr_evidential_support[index, feature_id]
        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0
        loges = np.log(evidential_support)  # 取自然对数，为了把乘法变成加法
        logunes = np.log(1 - evidential_support)
        evidential_support_logit = csr_matrix((loges, (row, col)), shape=(len(self.variables), len(self.features)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)), shape=(len(self.variables), len(self.features)))
        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))

        approximate_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)
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

    def get_dict_rel_acc(self):
        '''
        get the accuracy of different types of relations
        :return:
        '''
        relations_name = set()
        relations_id = set()
        dict_rel_acc = dict()
        for feature in self.features:
            feature_id = feature['feature_id']
            if feature['feature_type'] == 'binary_feature':
                relations_id.add(feature_id)
                relations_name.add(feature['feature_name'])
        dict_reltype_edges = {rel: list() for rel in relations_name}
        for fid in relations_id:
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
        # l: support for labeling
        # u: support for unalbeling
        '''
        return MassFunction({'l': (1 - uncertain_degree) * label_prob,
                             'u': (1 - uncertain_degree) * unlabel_prob,
                             'lu': uncertain_degree})

    def labeling_propensity_with_ds(self,mass_functions):
        combined_mass = gml_utils.combine_evidences_with_ds(mass_functions, normalization=True)
        return combined_mass

    def evidential_support_by_relation(self,variable_set,update_feature_set=None):
        '''
        计算给定隐变量集合中每个隐变量的evidential support,适用于ALSA
        :param update_feature_set:
        :param variable_set:
        :return:
        '''
        self.get_unlabeled_var_feature_evi()
        if type(variable_set) == list or type(variable_set) == set:
            mass_functions = list()
            for vid in variable_set:
                if len(self.variables[vid]['binary_feature_evi']) > 0:
                    for (anotherid, feature_name, feature_id) in self.variables[vid]['binary_feature_evi']:
                        # (anotherid,feature_name,feature_id)
                        rel_acc = self.dict_rel_acc[feature_name]
                        mass_functions.append(self.construct_mass_function_for_propensity(self.relation_evi_uncer_degree,rel_acc, 1-rel_acc))
                if len(self.variables[vid]['unary_feature_evi_prob']) > 0:
                    for (feature_id, feature_name, n_samples, neg_prob, pos_prob) in self.variables[vid]['unary_feature_evi_prob']:
                        # feature_id,feature_name,n_samples,neg_prob,pos_prob
                        mass_functions.append(self.construct_mass_function_for_propensity(self.word_evi_uncer_degree, max(pos_prob, neg_prob),min(pos_prob, neg_prob)))
                if len(mass_functions) > 0:
                    combine_evidential_support = self.labeling_propensity_with_ds(mass_functions)
                    self.variables[vid]['combine_evidential_support'] = combine_evidential_support
                    self.variables[vid]['evidential_support'] = combine_evidential_support['l']
                    mass_functions.clear()
                else:
                    self.variables[vid]['evidential_support'] = 0.0

            logging.info("evidential_support_by_relation finished")

    def evidential_support_by_custom(self,variable_set,update_feature_set=None):
        '''
        用户自定义计算evidential support
        :param update_feature_set:
        :return:
        '''
        pass

    def evidential_support_by_regression_binary(self,variable_set,update_feature_set, c = -1):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        # self.separate_feature_value_binary()
        if update_feature_set == None or (type(update_feature_set) ==set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid,feature in enumerate(self.features):
                update_feature_set.add(fid)
        self.influence_modeling_multi(variable_set, update_feature_set, c)
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []

        for fid in update_feature_set:
            feature = self.features[fid]
            if feature['feature_type'] == 'binary_feature':
                for k, v in self.features[fid]['weight'].items():
                    if 'regression' in feature.keys():
                        if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][
                            0] >= 0:
                            feature['monotonicity'] = True
                        else:
                            feature['monotonicity'] = False

                        if feature['regression'].regression is not None and feature['regression'].variance > 0:
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
                        variance.append(
                            feature['regression'].variance if feature['regression'].variance is not None else np.NaN)

        coefs = np.array(coefs)[col]
        intercept = np.array(intercept)[col]
        zero_confidence = np.array(zero_confidence)[col]
        residuals, Ns, meanX, variance = np.array(residuals)[col], np.array(Ns)[col], np.array(meanX)[col], \
                                         np.array(variance)[col]

        temp = residuals * np.sqrt(1 + 1.0 / Ns + np.power(data - meanX, 2) / variance)
        temp[temp <= 0.0] = 0.0001
        tvalue = float(delta) / (temp)

        confidence = np.ones_like(data)
        confidence[np.where(residuals > 0)] = (1 - t.sf(tvalue, (Ns - 2)) * 2)[np.where(residuals > 0)]
        confidence = confidence * zero_confidence
        evidential_support = (1 + confidence) / 2  # 正则化
        # 将计算出的evidential support写回
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.variables)))


        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence

        approximate_weight = csr_matrix((espredict, (row, col)),
                                        shape=(len(self.variables), len(self.variables)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)


        ####################################################################################################################
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0

        loges = np.log(evidential_support)  # 取自然对数，为了把乘法变成加法

        temp_arr = 1 - evidential_support
        # temp_arr[temp_arr <= 0.0] = 0.0001
        logunes = np.log(temp_arr)

        evidential_support_logit = csr_matrix((loges, (row, col)),
                                              shape=(len(self.variables), len(self.variables)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)),
                                                shape=(len(self.variables), len(self.variables)))


        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        ####################################################################################################################
        # for vid in range(self.variables):
        #     self.variables[vid]['binary_weight'] = approximate_weight[vid]
        #
        #     evidential_support_binary = self.variables[vid].get('evidential_support_binary', 0.0)
        #     evidential_support_binary += float(p_es[vid] / (p_es[vid] + p_unes[vid]))
        #     self.variables[vid]['evidential_support_binary'] = evidential_support_binary

        # # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            approximate_weight_binary = var.get('approximate_weight_binary', 0.0)
            approximate_weight_binary += approximate_weight[index]
            var['approximate_weight_binary'] = approximate_weight_binary
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]

            evidential_support_binary = self.variables[index].get('evidential_support_binary', 0.0)
            evidential_support_binary += float(var_p_es / (var_p_es + var_p_unes))
            self.variables[index]['evidential_support_binary'] = evidential_support_binary

        # # 近似权重写回
        # csr_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.variables)))
        # for fid in update_feature_set:
        #     # 双因子只有一组变量，所以weight只有一组数值
        #     for k, v in self.features[fid]['weight'].items() :
        #         first = int(k[0])
        #         second = int(k[1])
        #         v[0] = csr_weight[first, second]


    def evidential_support_by_regression_single(self,variable_set,update_feature_set, c):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        # self.separate_feature_value_single(c)
        if update_feature_set == None or (type(update_feature_set) ==set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid,feature in enumerate(self.features):
                update_feature_set.add(fid)
        self.influence_modeling_multi(variable_set, update_feature_set, c)
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []
        # 单因子这部分是需要改成多分类，即按维度取值
        for feature in self.features:
            if 'regression' in feature.keys():

                if feature['feature_type'] == 'unary_feature':
                    if feature['regression'].get(c).regression is not None and feature['regression'].get(c).regression.coef_[0][0] >=0:
                        if 'monotonicity' in feature.keys():
                            feature['monotonicity'][c] = True
                        else:
                            feature['monotonicity'] = [False] * self.classNum
                            feature['monotonicity'][c] = True
                    else:
                        if 'monotonicity' in feature.keys():
                            feature['monotonicity'][c] = False
                        else:
                            feature['monotonicity'] = [False] * self.classNum
                            feature['monotonicity'][c] = False
                    if feature['regression'].get(c).regression is not None and feature['regression'].get(c).variance > 0:
                        coefs.append(feature['regression'].get(c).regression.coef_[0][0])
                        intercept.append(feature['regression'].get(c).regression.intercept_[0])
                        zero_confidence.append(1)
                    else:
                        coefs.append(0)
                        intercept.append(0)
                        zero_confidence.append(0)
                    Ns.append(feature['regression'].get(c).N if feature['regression'].get(c).N > feature[
                        'regression'].get(c).effective_training_count else np.NaN)
                    residuals.append(feature['regression'].get(c).residual if feature['regression'].get(c).residual is not None else np.NaN)
                    meanX.append(feature['regression'].get(c).meanX if feature['regression'].get(c).meanX is not None else np.NaN)
                    variance.append(feature['regression'].get(c).variance if feature['regression'].get(c).variance is not None else np.NaN)
                # 之所以计算单因子也把双因子加进来，是为了对齐矩阵
                else:
                    feature['monotonicity'] = False
                    coefs.append(0)
                    intercept.append(0)
                    zero_confidence.append(0)

                    Ns.append(np.NaN)
                    residuals.append(np.NaN)
                    meanX.append(np.NaN)
                    variance.append(np.NaN)

            else:
                feature['monotonicity'] = False
                coefs.append(0)
                intercept.append(0)
                zero_confidence.append(0)

                Ns.append(np.NaN)
                residuals.append(np.NaN)
                meanX.append(np.NaN)
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
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.features)))
        for index, var in enumerate(self.variables):
            for fid in var['feature_set']:
                if fid in update_feature_set:
                    var['feature_set'][fid][0][c] = csr_evidential_support[index, fid]

        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence

        approximate_weight = csr_matrix((espredict, (row, col)),
                                        shape=(len(self.variables), len(self.variables)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)

        ####################################################################################################################
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0
        loges = np.log(evidential_support)  # 取自然对数，为了把乘法变成加法
        logunes = np.log(1 - evidential_support)
        evidential_support_logit = csr_matrix((loges, (row, col)),
                                              shape=(len(self.variables), len(self.variables)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)),
                                                shape=(len(self.variables), len(self.variables)))

        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        ####################################################################################################################

        # # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            if 'approximate_weight' not in var.keys():
                var['approximate_weight'] = [0] * self.classNum
            var['approximate_weight'][c] = approximate_weight[index]
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            if 'evidential_support' not in self.variables[index].keys():
                self.variables[index]['evidential_support'] = [0] * self.classNum
            self.variables[index]['evidential_support'][c] = float(var_p_es / (var_p_es + var_p_unes))

        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        # for var_id in self.poential_variables_set:
        #     index = var_id
        #     var_p_es = p_es[index]
        #     var_p_unes = p_unes[index]
        #     if 'evidential_support' not in self.variables[index].keys():
        #         self.variables[index]['evidential_support'] = [0] * self.classNum
        #     self.variables[index]['evidential_support'][c] = float(var_p_es / (var_p_es + var_p_unes))

        # 近似权重写回
        csr_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))
        for fid in update_feature_set:
            for k, v in self.features[fid]['weight'].items():
                v[0][c] = csr_weight[k, fid]
        #
        #         if 'approximate_weight' not in self.variables[first].keys():
        #             self.variables[first]['approximate_weight'] = [0.0] * self.classNum
        #         self.variables[first]['approximate_weight'][c] += approximate_weight[first]

    def evidential_support_by_regression_distance(self,variable_set,update_feature_set):
        '''
        计算给定隐变量集合的Evidential Support,适用于ER
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        # self.separate_feature_value_single(c)
        if update_feature_set == None or (type(update_feature_set) ==set() and len(update_feature_set) == 0):
            update_feature_set = set()
            for fid,feature in enumerate(self.features):
                update_feature_set.add(fid)
        self.influence_modeling(update_feature_set)
        coo_data = self.data_matrix.tocoo()
        row, col, data = coo_data.row, coo_data.col, coo_data.data
        coefs = []
        intercept = []
        residuals = []
        Ns = []
        meanX = []
        variance = []
        delta = self.delta
        zero_confidence = []

        for fid in update_feature_set:
            feature = self.features[fid]
            if feature['feature_name'] == 'distance_feature':
                for k, v in self.features[fid]['weight'].items():
                    if 'regression' in feature.keys():
                        if feature['regression'].regression is not None and feature['regression'].regression.coef_[0][
                            0] >= 0:
                            feature['monotonicity'] = True
                        else:
                            feature['monotonicity'] = False

                        if feature['regression'].regression is not None and feature['regression'].variance > 0:
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
                        variance.append(
                            feature['regression'].variance if feature['regression'].variance is not None else np.NaN)

        temp = coefs
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
        csr_evidential_support = csr_matrix((evidential_support, (row, col)),
                                            shape=(len(self.variables), len(self.features)))
        for index, var in enumerate(self.variables):
            for fid in var['feature_set']:
                if fid in update_feature_set:
                    var['feature_set'][fid][0] = csr_evidential_support[index, fid]

        # 计算近似权重
        predict = data * coefs + intercept
        espredict = predict * evidential_support
        # espredict[np.where(polar_enforce == 0)] = np.minimum(espredict, 0)[np.where(polar_enforce == 0)]   #小于0就变成0
        # espredict[np.where(polar_enforce == 1)] = np.maximum(espredict, 0)[np.where(polar_enforce == 1)]   #大于0就变成1
        espredict = espredict * zero_confidence

        approximate_weight = csr_matrix((espredict, (row, col)),
                                        shape=(len(self.variables), len(self.variables)))
        approximate_weight = np.array(approximate_weight.sum(axis=1)).reshape(-1)

        ####################################################################################################################
        # 验证所有evidential support都在(0,1)之间
        assert len(np.where(evidential_support < 0)[0]) == 0
        assert len(np.where((1 - evidential_support) < 0)[0]) == 0
        loges = np.log(evidential_support)  # 取自然对数，为了把乘法变成加法
        logunes = np.log(1 - evidential_support)
        evidential_support_logit = csr_matrix((loges, (row, col)),
                                              shape=(len(self.variables), len(self.variables)))
        evidential_unsupport_logit = csr_matrix((logunes, (row, col)),
                                                shape=(len(self.variables), len(self.variables)))

        p_es = np.exp(np.array(evidential_support_logit.sum(axis=1)))
        p_unes = np.exp(np.array(evidential_unsupport_logit.sum(axis=1)))
        ####################################################################################################################

        # # 将计算出来的近似权重写回variables
        for index, var in enumerate(self.variables):
            var['approximate_weight_distance'] = approximate_weight[index]
        # 综合计算每个隐变量的evidential suppport,并写回self.variables
        for var_id in self.poential_variables_set:
            index = var_id
            var_p_es = p_es[index]
            var_p_unes = p_unes[index]
            self.variables[index]['evidential_support_distance'] = float(var_p_es / (var_p_es + var_p_unes))

        # 近似权重写回
        csr_weight = csr_matrix((espredict, (row, col)), shape=(len(self.variables), len(self.features)))
        for fid in update_feature_set:
            for k, v in self.features[fid]['weight'].items():
                v[0] = csr_weight[k, fid]
        #
        #         if 'approximate_weight' not in self.variables[first].keys():
        #             self.variables[first]['approximate_weight'] = [0.0] * self.classNum
        #         self.variables[first]['approximate_weight'][c] += approximate_weight[first]

    def approximate_weight_multi_origin(self):
        for c in range(self.classNum):
            for index, var in enumerate(self.variables):
                # 只计算隐变量
                if not var['is_evidence'] :
                    # print('类别: ', c)
                    # print('变量个数：', len(self.variables))
                    # print('变量feature个数：', len(var['feature_set'].keys()))
                    # print()

                    binary_es = var['evidential_support_binary'] if 'evidential_support_binary' in var.keys() else 0.0
                    binary_weight = var['approximate_weight_binary'] = var['evidential_support_binary'] if 'evidential_support_binary' in var.keys() else 0.0

                    single_es = var['evidential_support'][c]
                    single_weight = var['approximate_weight'][c]

                    # es 按照类别合并
                    if 'evidential_support' not in var.keys():
                        var['evidential_support'] = [0] * self.classNum

                    var['evidential_support'][c] = single_es + binary_es
                    # var['evidential_support'][c] = var['evidential_support'][c] + binary_es

                    if 'approximate_weight' not in var.keys():
                        var['approximate_weight'] = [0] * self.classNum

                    var['approximate_weight'][c] = single_weight + binary_weight

    def approximate_weight_multi(self):

        # for fea in self.features:
        #     if fea['feature_type'] == 'unary_feature':
        #         fea['tau'] = [0.5] * self.classNum
        #         fea['alpha'] = [0.0] * self.classNum
        #     if fea['feature_type'] == 'binary_feature':
        #         fea['tau'] = 0.5
        #         fea['alpha'] = 0.0

        for c in range(self.classNum):
            approximate_weight = list()
            for index, var in enumerate(self.variables):
                # 只计算隐变量
                if not var['is_evidence'] :
                    var_id = var['var_id']

                    single_weight = 0
                    binary_weight = 0
                    for fea_id in var['feature_set']:
                        if self.features[fea_id]['feature_type'] == 'unary_feature':
                            # fea_value = self.features[fea_id]['weight'][var_id][1]
                            # tau = self.features[fea_id]['tau'][c]
                            # alpha = self.features[fea_id]['alpha'][c]
                            # single_weight += var['evidential_support'] * tau * (fea_value - alpha)
                            # self.features[fea_id]['weight'][var_id][0][c] = single_weight
                            if self.features[fea_id]['Association_category'] == c:
                                single_weight += self.features[fea_id]['weight'][var_id][0]
                        else:
                            # tau = self.features[fea_id]['tau']
                            # alpha = self.features[fea_id]['alpha']
                            for (varId1, varId2) in self.features[fea_id]['weight']:
                                if (var_id == varId1 or var_id == varId2):
                                    otherVarId = varId1 if var_id == varId2 else varId2
                                    if (self.variables[otherVarId]['is_evidence'] == True and self.variables[otherVarId]['label'] == c):
                                        # fea_value = self.features[fea_id]['weight'][(varId1, varId2)][1]
                                        # binary_weight += var['evidential_support'] * tau * (fea_value - alpha)
                                        # self.features[fea_id]['weight'][(varId1, varId2)][0] = binary_weight
                                        binary_weight += self.features[fea_id]['weight'][(varId1, varId2)][0]

                    if 'approximate_weight' not in var.keys():
                        var['approximate_weight'] = [0] * self.classNum
                    var['approximate_weight'][c] = single_weight + binary_weight
                    approximate_weight.append(var['approximate_weight'][c])
            # print("over")
            # print("最大approximate_weight：", max(approximate_weight))
            # print("最小approximate_weight：", min(approximate_weight))
            # print()

        print("权重合并over！")

    def evidential_support_by_multi_(self,variable_set,update_feature_set=None):
        # 这个版本是单双因子都没有线性回归，es的计算只取决于连接的显变量
        def MaxMinNormalization(x, Max, Min):
            division = Max - Min
            if division < 1e-6:
                division = 1e-6
            res = (float)(x - Min) / division
            return res
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        binaryCountSet = set()
        binaryCountSet.add(0)
        for feature in self.features:
            if feature['feature_type'] == 'binary_feature' and feature['feature_name'] == 'BERT_CosineSimilarity':
                for k in feature['weight'].keys():
                    if k[0] in self.poential_variables_set:
                        if k[1] in self.observed_variables_set:
                            temp = self.variables[k[0]].get('binary_count', 0) + 1
                            self.variables[k[0]]['binary_count'] = temp
                            binaryCountSet.add(temp)

                    if k[1] in self.poential_variables_set:
                        if k[0] in self.observed_variables_set:
                            temp = self.variables[k[1]].get('binary_count', 0) + 1
                            self.variables[k[1]]['binary_count'] = temp
                            binaryCountSet.add(temp)

        Max = max(binaryCountSet)
        Min = min(binaryCountSet)
        for vid in self.poential_variables_set:
            self.variables[vid]['evidential_support_binary'] = MaxMinNormalization(self.variables[vid].get('binary_count', 0), Max, Min)
            self.variables[vid]['binary_count'] = 0

        # single
        singleCountSet = set()
        singleCountSet.add(0)
        for vid in self.poential_variables_set:
            count = 0
            for fid in self.variables[vid]['feature_set'].keys():
                if self.features[fid]['feature_type'] == 'unary_feature':
                    for k in self.features[fid]['weight'].keys():
                        if self.variables[k]['is_evidence']:
                            count += 1
            singleCountSet.add(count)
            self.variables[vid]['single_count'] = count

        Max = max(singleCountSet)
        Min = min(singleCountSet)
        es = list()
        for vid in self.poential_variables_set:
            self.variables[vid]['evidential_support_single'] = MaxMinNormalization(self.variables[vid]['single_count'], Max, Min)
            self.variables[vid]['single_count'] = 0
            self.variables[vid]['evidential_support'] = (self.variables[vid]['evidential_support_binary'] +
                                                       self.variables[vid]['evidential_support_single'] + 1) / 2.0

            es.append(self.variables[vid]['evidential_support'])

        # print("over")
        # print("最大es：", max(es))
        # print("最小es：", min(es))

        print("合并权重开始：")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print()
        self.approximate_weight_multi()
        print("合并权重结束：")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print()

    def evidential_support_by_multi(self,variable_set,update_feature_set=None):
        # 双因子已经线性回归
        def MaxMinNormalization(x, Max, Min):
            division = Max - Min
            if division < 1e-6:
                division = 1e-6
            res = (float)(x - Min) / division
            return res
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)


        # binary
        # # 存放features的所有easy的featurevalue   :feature_id:[[value1,bound],[value2,bound]...]
        features_easys_binary = dict()
        # 存放参与线性回归的因子，只有这部分因子参与计算。
        update_feature_binary_set = set()
        # 存放参与计算的变量
        variable_binary_set = set()
        # 遍历所有因子
        for feature in self.features:

            fid = feature['feature_id']

            # 找到双因子，确定正样本和负样本
            if feature["feature_type"] == "binary_feature":
                for k, value in feature["weight"].items():
                    temp0 = int(k[0])
                    temp1 = int(k[1])
                    variable_binary_set.add(temp0)
                    variable_binary_set.add(temp1)
                    update_feature_binary_set.add(fid)

                    if self.variables[temp0]['is_evidence'] and self.variables[temp1]['is_evidence']:
                        # 两篇文档同类
                        if self.variables[temp0]['label'] == self.variables[temp1]['label']:
                            if fid in features_easys_binary.keys():
                                features_easys_binary[fid].append([value[1], 1])
                            else:
                                features_easys_binary[fid] = list()
                                features_easys_binary[fid].append([value[1], 1])
                        else:
                            if fid in features_easys_binary.keys():
                                features_easys_binary[fid].append([value[1], 0])
                            else:
                                features_easys_binary[fid] = list()
                                features_easys_binary[fid].append([value[1], 0])

        self.features_easys = features_easys_binary
        self.data_matrix = self.create_csr_matrix_binary(variable_binary_set, update_feature_binary_set)
        self.evidential_support_by_regression_binary(variable_binary_set, update_feature_binary_set)

        # # 新加基于距离的单因子
        # # # 存放features的所有easy的featurevalue   :feature_id:[[value1,bound],[value2,bound]...]
        # features_easys_distance = dict()
        # # 存放参与线性回归的因子，只有这部分因子参与计算。
        # update_feature_distance_set = set()
        # # 存放参与计算的变量
        # variable_distance_set = set()
        # # 遍历所有因子
        # for feature in self.features:
        #     fid = feature['feature_id']
        #     # 找到距离单因子，确定正样本和负样本
        #     if feature["feature_name"] == "distance_feature":
        #         for k, value in feature["weight"].items():
        #             variable_distance_set.add(k)
        #             update_feature_distance_set.add(fid)
        #             if self.variables[k]['is_evidence']:
        #                 if fid not in features_easys_distance.keys():
        #                     features_easys_distance[fid] = list()
        #                 # 如果文档确实是c类，则为正样本
        #                 if self.variables[k]['label'] == feature['Association_category']:
        #                     features_easys_distance[fid].append([value[1], 1])
        #                 # 否则是负样本
        #                 else:
        #                     features_easys_distance[fid].append([value[1], 0])
        # self.features_easys = features_easys_distance
        # self.data_matrix = self.create_csr_matrix_single(update_feature_distance_set)
        # self.evidential_support_by_regression_distance(variable_distance_set, update_feature_distance_set)

        # single
        singleCountSet = set()
        singleCountSet.add(0)
        for vid in self.poential_variables_set:
            count = 0
            for fid in self.variables[vid]['feature_set'].keys():
                if self.features[fid]['feature_type'] == 'unary_feature' : # and self.features[fid]['feature_name'] != 'distance_feature':
                    for k in self.features[fid]['weight'].keys():
                        if self.variables[k]['is_evidence']:
                            count += 1
            singleCountSet.add(count)
            self.variables[vid]['single_count'] = count

        Max = max(singleCountSet)
        Min = min(singleCountSet)
        es = list()
        for vid in self.poential_variables_set:
            self.variables[vid]['evidential_support_single'] = MaxMinNormalization(self.variables[vid]['single_count'], Max, Min)
            self.variables[vid]['single_count'] = 0
            evidential_support_diatance = 0.0 if 'evidential_support_diatance' not in self.variables[vid].keys() \
                else self.variables[vid]['evidential_support_diatance']
            self.variables[vid]['evidential_support'] = (self.variables[vid]['evidential_support_binary'] +
                                                       self.variables[vid]['evidential_support_single'] + 1 +
                                                         evidential_support_diatance) / 3.0

            es.append(self.variables[vid]['evidential_support'])

        # print("over")
        # print("最大es：", max(es))
        # print("最小es：", min(es))

        print("合并权重开始：")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print()
        self.approximate_weight_multi()
        print("合并权重结束：")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print()