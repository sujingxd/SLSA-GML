import heapq
import pickle
import sys
import time
from numbskull_extend import numbskull
import logging
from sklearn import metrics

from evidential_support_new import EvidentialSupport
from evidence_select import EvidenceSelect
from approximate_probability_estimation import ApproximateProbabilityEstimation
from construct_subgraph import ConstructSubgraph
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor
import gml_utils as gml_utils
from gml_utils import Logger
import numpy as np
import pandas as pd

class GML:
    '''
     GML main process: evidentail support->select_topm->ApproximateProbabilityEstimation->select->topk->inference->label->score
    '''
    def __init__(self, opt, variables, features, learning_method,top_m, top_k, top_n,update_proportion, balance,optimization_threshold,learning_epoches,inference_epoches ,nprocess,out):
        ## wyy
        self.opt = opt
        ## wyy
        #check data
        variables_keys = ['var_id','is_easy','is_evidence','true_label','label','feature_set']
        features_keys = ['feature_id','feature_type','parameterize','feature_name','weight']
        learning_methods = ['sgd', 'bgd']
        if learning_method not in learning_methods:
            raise ValueError('learning_methods has no this method: '+learning_method)
        ## wyy
        self.is_features_all_parameterize = True
        # for feature in features:
        #     for attribute in features_keys:
        #         if attribute not in feature:
        #             raise ValueError('features has no key: ' + attribute)
        #     if feature['parameterize'] == 0:
        #         self.is_features_all_parameterize = False
        ## wyy
        #check variables
        for variable in variables:
            for attribute in variables_keys:
                if attribute not in variable:
                    raise ValueError('variables has no key: '+attribute)
        #check features
        for feature in features:
            for attribute in features_keys:
                if attribute not in feature:
                    raise ValueError('features has no key: '+attribute)

        ## wyy
        ################################################################################################################
        self.all_iter_variables = []
        self.all_iter_features = []
        self.gradual_label_varids = []          ## 2021-05-29 [wyy]
        ################################################################################################################

        self.variables = variables
        self.features = features
        import copy
        self.ori_variables = copy.deepcopy(variables)
        self.ori_features = copy.deepcopy(features)
        self.learning_method = learning_method # now support sgd and bgd
        self.labeled_variables_set = set()
        self.top_m = top_m
        self.top_k = top_k
        self.top_n = top_n
        self.optimization_threshold = optimization_threshold #entropy optimization threshold: less than 0 means no need to optimize
        self.update_proportion = update_proportion #Evidence support update ratio: it is necessary to recalculate the evidence support after inferring a certain percentage of hidden variables
        self.support = EvidentialSupport(variables, features, opt)   ##wyy
        self.select = EvidenceSelect(variables, features)
        self.approximate = ApproximateProbabilityEstimation(variables,features)
        self.subgraph = ConstructSubgraph(variables, features, balance)
        self.learing_epoches = learning_epoches   #Factor graph parameter learning rounds
        self.inference_epoches = inference_epoches #Factor graph inference rounds
        self.nprocess = nprocess #Number of multiple processes
        self.out = out  #Do you need to record the results
        self.evidence_interval_count = 10  #Number of evidence intervals divided by featureValue   ## ??????
        self.all_feature_set = set([x for x in range(0,len(features))])   #Feature ID collection
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(variables)   #Dividing evidence variables and latent variables
        #Initialize the necessary properties
        self.evidence_interval = gml_utils.init_evidence_interval(self.evidence_interval_count)    #Divide the evidence interval   ## ?????
        gml_utils.init_bound(variables,features)   #Initial value of initialization parameter
        gml_utils.init_evidence(features,self.evidence_interval,self.observed_variables_set)
        self.train_ids = gml_utils.get_train_ids(variables)
        self.cnt = 0
        # self.virtual_ids = list(range(2262, 3016))
        # self.virtual_ids = list(range(5098, 7132))
        self.virtual_ids = list(range(255, 5097))
        # self.virtual_ids = list(range(6920, 8741))
        # self.virtual_ids = list(range(1248,))
        # self.virtual_ids = list(range(2096,2850))
        #save results
        self.now = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        self.result = self.now+'-result.txt'
        if self.out:
            with open(self.result, 'w') as f:
                f.write('var_id'+' '+'inferenced_probability'+' '+'inferenced_label'+' '+'ture_label'+'\n')
        #logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'
        )
        logging.info("GML inference begin")

    @staticmethod
    def initial(configFile, opt, variables, features):
        '''
        load config from file
        @param configFile:
        @param variables:
        @param features:
        @return:
        '''
        config = ConfigParser()
        #Set default parameters
        config.read_dict({'para':{'learning_method':'sgd',
                                  'learning_epoches':'1000',
                                  'inference_epoches':'1000',
                                  'top_m':'200',
                                  'top_k':'5',
                                  'top_n':'5',
                                  'n_process':'1',
                                  'update_proportion':'0.1',
                                  'balance':'True',## False
                                  'optimization_threshold':'1', # 1
                                  'out':'True'}
                          })
        config.read(configFile, encoding='UTF-8')
        learning_method = config['para']['learning_method']
        learning_epoches = int(config['para']['learning_epoches'])
        inference_epoches = int(config['para']['inference_epoches'])
        top_m = int(config['para']['top_m'])
        top_k = int(config['para']['top_k'])
        top_n = int(config['para']['top_n'])
        n_process = int(config['para']['n_process'])
        update_proportion = float(config['para']['update_proportion'])
        balance = config['para'].getboolean('balance')
        optimization_threshold = float(config['para']['optimization_threshold'])
        out = config['para'].getboolean('out')
        return GML(opt, variables, features, learning_method, top_m, top_k, top_n,update_proportion,balance,optimization_threshold,learning_epoches,inference_epoches,n_process,out)

    def evidential_support(self, variable_set, update_feature_set):
        '''
        calculate evidential_support
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        self.support.evidential_support(variable_set, update_feature_set, self.train_ids)

    def approximate_probability_estimation(self, variable_set):
        '''
        estimation approximate_probability
        @param variable_set:
        @return:
        '''
        if self.is_features_all_parameterize == False:
            self.approximate.approximate_probability_estimation(variable_set)
        else:
            self.approximate.approximate_probability_estimation_by_interval(variable_set)  # 2034

    def select_top_m_by_es(self, m):
        '''
        select tom m largest ES poential variables
        @param m:
        @return: m_id_list
        '''
        #If the current number of hidden variables is less than m, directly return to the hidden variable list
        if m > len(self.poential_variables_set):
           return list(self.poential_variables_set)
        poential_var_list = list()
        m_id_list = list()
        observed_var_list = list()
        for var_id in self.poential_variables_set:
            poential_var_list.append([var_id, self.variables[var_id]['evidential_support']])
        #Select the evidence to support the top m largest
        topm_var = heapq.nlargest(m, poential_var_list, key=lambda s: s[1])
        for elem in topm_var:
            m_id_list.append(elem[0])
        logging.info('select m finished')

        ## for print train prob
        # for var_id in self.observed_variables_set:
        #     observed_var_list.append([var_id, self.variables[var_id]['evidential_support']])
        # out_train_prob = pd.DataFrame()
        # out_train_prob['id'] = [id for id, prob in observed_var_list]
        # out_train_prob['prob'] = [prob for id, prob in observed_var_list]
        # id_to_true_labels = dict()
        # true_labels = []
        # for each_unit in self.variables:
        #     id_to_true_labels[each_unit['var_id']] = each_unit['true_label']
        # for id in list(out_train_prob['id']):
        #     true_labels.append(id_to_true_labels[id])
        # out_train_prob.to_csv('train_prob_out.csv', sep='\t', index=None)

        return m_id_list

    def select_top_k_by_entropy(self, var_id_list, k):
        '''
        select top k  smallest entropy poential variables
        @param var_id_list:
        @param k:
        @return:
        '''
        #If the number of hidden variables is less than k, return the hidden variable list directly
        if len(var_id_list) < k:
            return var_id_list
        m_list = list()
        k_id_list = list()
        for var_id in var_id_list:
            m_list.append(self.variables[var_id])
        #Pick the top k with the smallest entropy
        k_list = heapq.nsmallest(k, m_list, key=lambda x: x['entropy'])
        for var in k_list:
            k_id_list.append(var['var_id'])
        logging.info('select k finished')
        return k_id_list

    def evidence_select(self, var_id):
        '''
        Determine the subgraph structure
        @param var_id:
        @return:
        '''
        connected_var_set, connected_edge_set, connected_feature_set = self.select.evidence_select(var_id)
        return connected_var_set, connected_edge_set, connected_feature_set

    def construct_subgraph(self, var_id):
        '''
        Construct subgraphs according to numbskull requirements
        @param var_id:
        @return:
        '''
        evidences = self.evidence_select(var_id)
        weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map_feature,sample_list,wmap,wfactor = self.subgraph.construct_subgraph(evidences,var_id)
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map,alpha_bound,tau_bound,weight_map_feature,sample_list,wmap,wfactor


    def inference_subgraph(self, var_id):
        '''
        Subgraph parameter learning and reasoning
        @param var_id:
        @return:
        '''
        if not type(var_id) == int:
            raise ValueError('var_id should be int' )
        ns_learing = numbskull.NumbSkull(
            n_inference_epoch=self.learing_epoches,
            n_learning_epoch=self.inference_epoches,
            stepsize=0.01,
            decay=0.95,
            reg_param=1e-6,
            regularization=2,
            truncation=10,
            quiet=(not False),
            verbose=False,
            learn_non_evidence=True,
            sample_evidence=False,
            burn_in=10,
            nthreads=1,
            learning_method = self.learning_method
        )
        weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map_feature, sample_list, wmap, wfactor = \
            self.construct_subgraph(var_id)
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num, alpha_bound, tau_bound, sample_list, wmap, wfactor
        ns_learing.loadFactorGraph(*subgraph)
        # parameter learning
        ns_learing.learning()
        logging.info("subgraph learning finished")
        # Control the range of the learned parameters and set the isfixed attribute of weight to true
        for index, w in enumerate(weight):
            feature_id = weight_map_feature[index]
            w["isFixed"] = True
            if self.features[feature_id]['parameterize'] == 1:
                theta = self.variables[var_id]['feature_set'][feature_id][0]
                x = self.variables[var_id]['feature_set'][feature_id][1]
                a = ns_learing.factorGraphs[0].weight[index]['a']
                b = ns_learing.factorGraphs[0].weight[index]['b']
                w['initialValue'] = theta * a * (x - b)
            else:
                w['initialValue'] = ns_learing.factorGraphs[0].poential_weight[index]

        ns_inference = numbskull.NumbSkull(
            n_inference_epoch=self.learing_epoches,
            n_learning_epoch= self.inference_epoches,
            stepsize=0.001,
            decay=0.95,
            reg_param=1e-6,
            regularization=2,
            truncation=10,
            quiet=(not False),
            verbose=False,
            learn_non_evidence=False,
            sample_evidence=False,
            burn_in=10,
            nthreads=1,
            learning_method=self.learning_method
        )
        ns_inference.loadFactorGraph(*subgraph)
        ns_inference.inference()
        logging.info("subgraph inference finished")
        # Write back the probability to self.variables
        if type(var_id) == set or type(var_id) == list:
            for id in var_id:
                self.variables[id]['inferenced_probability'] = ns_inference.factorGraphs[0].marginals[var_map[id]]
        elif type(var_id) == int:
            self.variables[var_id]['inferenced_probability'] = ns_inference.factorGraphs[0].marginals[var_map[var_id]]
        logging.info("inferenced probability recored")

    def label(self, var_id_list, isapprox, mlist=None):
        '''
        Select n from k inferred hidden variables for labeling
        @param var_id_list:
        @return:
        '''
        entropy_list = list()
        label_list = list()
        if isapprox == True:
            probability_indicator = 'approximate_probability'
        else:
            probability_indicator = 'inferenced_probability'
        #Calculate the entropy of k hidden variables
        for var_id in var_id_list:
            var_index = var_id
            self.variables[var_index]['entropy'] = gml_utils.entropy(self.variables[var_index][probability_indicator])  # 拿近似概率计算出来的熵
            entropy_list.append([var_id, self.variables[var_index]['entropy']])
        #If labelnum is less than the number of variables passed in, mark top_n
        if len(var_id_list) > self.top_n:
            var = list()
            min_var_list = heapq.nsmallest(self.top_n, entropy_list, key=lambda x: x[1])  # 选出熵最小的变量
            for mv in min_var_list:
                label_list.append(mv[0])
        else:
            label_list = var_id_list
        for var_index in label_list:
            self.variables[var_index]['probability'] = self.variables[var_index][probability_indicator]
            self.variables[var_index]['label'] = 1 if self.variables[var_index]['probability'] >= 0.5 else 0
            # if var_index not in self.virtual_ids:
            #     self.variables[var_index]['label'] = self.variables[var_index]['true_label']
            self.variables[var_index]['is_evidence'] = True
            logging.info('var-' + str(var_index) + " labeled succeed--------------------------------------")
            self.poential_variables_set.remove(var_index)
            self.observed_variables_set.add(var_index)
            self.labeled_variables_set.add(var_index)
            probability = self.variables[var_index]['probability']
            label = self.variables[var_index]['label']
            true_label = self.variables[var_index]['true_label']
            if self.out:
                with open(self.result, 'a') as f:
                    f.write(f'{var_index:7} {probability:10} {label:4} {true_label:4}')
                    f.write('\n')
            ## 2021-5-29 wyy
            ############################################################################################################
            self.gradual_label_varids.append(var_index)
        gml_utils.update_evidence(self.variables, self.features, label_list, self.evidence_interval)
            ############################################################################################################
        return len(label_list)


    # def update_slidingWindow_factor(self):
    #     self.vaname2classname_fv.clear()
    #     for eachvarname in self.varname2isevidence:
    #         self.compute_slidingWindow_featureValue(eachvarname)
    #     # print(self.vaname2classname_fv)
    #     print('compute sliding window finished ')
        # self.prior_probability()


    def update_slidingWindow_factor(self):
        varname_list = []
        neighbor_pos_list = []
        neighbor_neg_list = []
        fv_list = []
        all_true_labels = []
        pre_label = []
        for eachvar in self.variables:
            self.compute_slidingWindow_featureValue(eachvar['var_id'], 1)
            # varname, neighbor_pos, neighbor_neg, fv = self.compute_slidingWindow_featureValue(eachvar['var_id'], 1)
            # varname_list.append(varname)
            # neighbor_pos_list.append(neighbor_pos)
            # neighbor_neg_list.append(neighbor_neg)
            # fv_list.append(fv)
            # if eachvar['var_id']>=226:
            #     all_true_labels.append(eachvar['true_label'])
            #     pre_label.append(1 if fv >0.5 else 0)

        self.cnt += 1
        print('compute sliding window finished ')
        # import pandas as pd
        # pdf = pd.DataFrame()
        # pdf['varname'] = varname_list
        # pdf['neighbor_pos'] = neighbor_pos_list
        # pdf['neighbor_neg'] = neighbor_neg_list
        # pdf['fv'] = fv_list
        # pdf.to_csv('result_{}'.format(self.cnt), index=None)
        # print('this is {}'.format(self.cnt))
        # print(metrics.accuracy_score(all_true_labels,pre_label))

    def compute_slidingWindow_featureValue(self, varname,windows_feature_id):
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        train_neighbors_cnt = 0
        test_neighbors_cnt = 0
        neighbor_pos = 0
        neighbor_neg = 0
        for each_feature_id in self.ori_variables[varname]['feature_set'].keys():
            if each_feature_id!= windows_feature_id:
                continue
            neighbors_pair_list = self.ori_variables[varname]['feature_set'][each_feature_id]
            for each_neighbors_pair in neighbors_pair_list:
                left_id = each_neighbors_pair[0]
                right_id = each_neighbors_pair[1]
                if left_id == varname:
                    other_id = right_id
                else:
                    other_id = left_id
                if other_id in self.train_ids:
                   train_neighbors_cnt += 1
                if other_id not in self.train_ids:
                    test_neighbors_cnt += 1
                if self.variables[other_id]['label'] == 0 and self.variables[other_id]['label'] != -1:
                   neighbor_neg += 1
                elif self.variables[other_id]['label'] == 1 and self.variables[other_id]['label'] != -1:
                   neighbor_pos += 1
                else:
                   pass
                if train_neighbors_cnt >3:
                    break
            if each_feature_id != 0:
                if neighbor_pos + neighbor_neg == 0:
                    fv = 0.5
                    self.features[each_feature_id]['weight'][varname] = [0.5, fv]
                    self.variables[varname]['feature_set'][each_feature_id] = [0.5, fv]
                else:
                    fv = neighbor_pos/(neighbor_pos + neighbor_neg)
                    self.features[each_feature_id]['weight'][varname] = [0.5 if fv>0.5 else -0.5, fv ]
                    self.variables[varname]['feature_set'][each_feature_id] = [0.5 if fv>0.5 else -0.5, fv ]


    def padding_slidingWindow_factor(self,train_ids, windows_feature_id):
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(self.variables)
        # for var_id in list(self.observed_variables_set)+list(self.poential_variables_set):
        for var_id in list(self.observed_variables_set)+list(self.poential_variables_set):
            train_neighbors_cnt = 0
            test_neighbors_cnt = 0
            neighbor_pos = 0
            neighbor_neg = 0
            for each_feature_id in self.ori_variables[var_id]['feature_set'].keys():
                if windows_feature_id!= each_feature_id:
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
                    if other_id in self.observed_variables_set:
                        if self.variables[other_id]['label'] == 0 and self.variables[other_id]['label'] != -1:
                            neighbor_neg += 1
                        elif self.variables[other_id]['label'] == 1  and self.variables[other_id]['label'] != -1:
                            neighbor_pos += 1
                        else:
                            pass
                    if train_neighbors_cnt >3:
                        break
            if each_feature_id!=0:
                if neighbor_pos + neighbor_neg == 0:
                    fv = 0.5
                    self.features[each_feature_id]['weight'][var_id] = [0.5, fv]
                    self.variables[var_id]['feature_set'][each_feature_id] = [0.5, fv]
                else:
                    fv = neighbor_pos/(neighbor_pos + neighbor_neg)
                    self.features[each_feature_id]['weight'][var_id] = [0.5 if fv >0.5 else -0.5, fv]
                    self.variables[var_id]['feature_set'][each_feature_id] = [0.5 if fv >0.5 else -0.5, fv]
        import copy
        tmp = copy.deepcopy(self.features)
        for each_pair in self.features[each_feature_id]['weight']:
            if type(each_pair) == tuple:
                del tmp[each_feature_id]['weight'][each_pair]
        self.features = tmp
        tmp_var = self.variables


    def inference(self):
        '''
        Through the main process
        @return:
        '''
        labeled_var = 0
        labeled_count = 0
        update_feature_set = set()  # Stores features that have changed during a round of updates
        inferenced_variables_id = set()  # Hidden variables that have been established and inferred during a round of update
        pool = ProcessPoolExecutor(self.nprocess)
        self.padding_slidingWindow_factor(self.train_ids, 1)
        if self.update_proportion > 0:
            update_cache = int(self.update_proportion * len(
                self.poential_variables_set))  # Evidential support needs to be recalculated every time update_cache variables are inferred
        self.update_slidingWindow_factor()
        self.evidential_support(self.poential_variables_set, self.all_feature_set)
        # self.approximate_probability_estimation(self.poential_variables_set)
        self.approximate_probability_estimation(self.poential_variables_set.symmetric_difference(self.observed_variables_set))
        m_list = self.select_top_m_by_es(self.top_m)
        while len(self.poential_variables_set) > 0:
            if self.update_proportion > 0 and labeled_var >= update_cache:
                # When the number of marked variables reaches update_cache, re-regression and calculate the emergency support
                for var_id in self.labeled_variables_set:
                    for feature_id in self.variables[var_id]['feature_set'].keys():
                        update_feature_set.add(feature_id)
                self.update_slidingWindow_factor()
                self.evidential_support(self.poential_variables_set, update_feature_set)
                self.approximate_probability_estimation(self.poential_variables_set)
                labeled_var = 0
                update_feature_set.clear()
                self.labeled_variables_set.clear()
                inferenced_variables_id.clear()
                # m_list = self.select_top_m_by_es(self.top_m)
            # elif len(m_list) == 0:
                # m_list = self.select_top_m_by_es(self.top_m)
            m_list = self.select_top_m_by_es(self.top_m)
            k_list = self.select_top_k_by_entropy(m_list, self.top_k)
            approx_list = []
            add_list = [x for x in k_list if
                        x not in inferenced_variables_id]  # Added variables in each round of reasoning
            if (self.nprocess == 1):
                for var_id in k_list:
                    # If the entropy is less than a certain threshold, mark it directly without reasoning
                    # edit min
                    if self.optimization_threshold == -2 or self.optimization_threshold >= 0 and self.variables[var_id][
                        'entropy'] <= self.optimization_threshold:
                        approx_list.append(var_id)
                if len(approx_list) > 0:
                    len_label_list = self.label(approx_list, isapprox=True)
                    k_list.clear()
                    labeled_var += len_label_list
                    labeled_count += len_label_list
                else:
                    for var_id in k_list:
                        if var_id in add_list:
                            self.inference_subgraph(var_id)
                            # For the variables that have been inferred during each round of update, because the parameters are not updated, there is no need for inference.
                            inferenced_variables_id.add(var_id)
            else:
                futures = []
                for var_id in add_list:
                    future = pool.submit(self.inference_subgraph, var_id)
                    futures.append(future)
                    inferenced_variables_id.add(var_id)
                for ft in futures:
                    self.variables[ft.result()[0]]['inferenced_probability'] = ft.result()[1]

            len_label_list = self.label(k_list, isapprox=False, mlist=m_list)  # edit at 2021-4-22
            labeled_var += len_label_list
            labeled_count += len_label_list
        self.score()

    def save_results(self):
        '''
        save results
        @return:
        '''
        with open(self.now+"_variables.pkl",'wb') as v:
            pickle.dump(self.variables, v)
        with open(self.now+"_features.pkl",'wb') as v:
            pickle.dump(self.features, v)

    # def trans(self, src_list):
    #     new_all_true_label = []
    #     for i in src_list:
    #         if i >= 0.5:
    #             new_all_true_label.append(1)
    #         else:
    #             new_all_true_label.append(0)
    #     return new_all_true_label
    #
    def score(self):
        '''
        Get results, including: accuracy, precision, recall, F1 score
        :return:
        '''
        easys_pred_label = list()
        easys_true_label = list()
        hards_pred_label = list()
        hards_true_label = list()
        for var in self.variables:
            if var['is_easy'] == True:
                easys_true_label.append(var['true_label'])
                easys_pred_label.append(var['label'])
            else:
                hards_true_label.append(var['true_label'])
                hards_pred_label.append(var['label'])
        all_true_label = easys_true_label + hards_true_label
        all_pred_label = easys_pred_label + hards_pred_label
        sys.stdout = Logger(self.result)
        print("--------------------------------------------")
        print("total:")
        print("--------------------------------------------")

        all_true_label = all_true_label
        all_pred_label = all_pred_label
        print("total accuracy_score: " + str(metrics.accuracy_score(all_true_label, all_pred_label)))
        print("total precision_score: " + str(metrics.precision_score(all_true_label, all_pred_label)))
        print("total recall_score: " + str(metrics.recall_score(all_true_label, all_pred_label)))
        print("total f1_score: " + str(metrics.f1_score(all_true_label, all_pred_label,average='macro')))
        print("--------------------------------------------")
        print("easys:")
        print("--------------------------------------------")
        easys_true_label = easys_true_label
        easys_pred_label = easys_pred_label
        print("easys accuracy_score:" + str(metrics.accuracy_score(easys_true_label, easys_pred_label)))
        print("easys precision_score:" + str(metrics.precision_score(easys_true_label, easys_pred_label)))
        print("easys recall_score:" + str(metrics.recall_score(easys_true_label, easys_pred_label)))
        # print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label,average='macro')))
        print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label)))
        print("--------------------------------------------")
        print("hards:")
        print("--------------------------------------------")
        hards_true_label = hards_true_label
        hards_pred_label = hards_pred_label
        print("hards accuracy_score:" + str(metrics.accuracy_score(hards_true_label, hards_pred_label)))
        print("hards precision_score: " + str(metrics.precision_score(hards_true_label, hards_pred_label)))
        print("hards recall_score: " + str(metrics.recall_score(hards_true_label, hards_pred_label)))
        print("hards f1_score: " + str(metrics.f1_score(hards_true_label, hards_pred_label)))
        # print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label)))

        observed_var_list = []
        for var_id in self.observed_variables_set:
            observed_var_list.append([var_id, self.variables[var_id]['approximate_probability']])
        out_train_prob = pd.DataFrame()
        out_train_prob['id'] = [id for id, prob in observed_var_list]
        out_train_prob['prob'] = [prob for id, prob in observed_var_list]
        id_to_true_labels = dict()
        train_true_labels = []
        train_pre_labels = []
        for each_unit in self.variables:
            id_to_true_labels[each_unit['var_id']] = each_unit['true_label']
        for id in list(out_train_prob['id']):
            train_true_labels.append(id_to_true_labels[id])
        for each_prob in list(out_train_prob['prob']):
            train_pre_labels.append(1 if each_prob >=0.5 else 0)
        out_train_prob['true_label'] = train_true_labels
        out_train_prob['pre_label'] = train_pre_labels
        out_train_prob.to_csv('train_prob_out.csv', sep='\t', index=None)

        print('train accuracy_score:'+ str(metrics.accuracy_score(train_true_labels, train_pre_labels)) )
        print("train f1_score: " + str(metrics.f1_score(train_true_labels, train_pre_labels,average='macro')))

    # def score(self):
    #         '''
    #         Get results, including: accuracy, precision, recall, F1 score
    #         :return:
    #         '''
    #         easys_pred_label = list()
    #         easys_true_label = list()
    #         hards_pred_label = list()
    #         hards_true_label = list()
    #         virtual_true_label = list()
    #         virtual_pre_label = list()
    #         easys_pred_ids = list()
    #         virtual_pre_ids = list()
    #         hards_pred_ids = list()
    #         virtual_ids = list(range(114, 2262)) # cr
    #         # virtual_ids = list(range(426, 8534)) # mr
    #         # virtual_ids = list(range(255, 5097))  # mr
    #         hard_cnt = 0
    #         easy_cnt = 0
    #         virtual_cnt = 0
    #         for var in self.variables:
    #             if var['is_easy'] == True:
    #                 easys_true_label.append(var['true_label'])
    #                 easys_pred_label.append(var['label'])
    #                 easy_cnt += 1
    #                 easys_pred_ids.append(var['var_id'])
    #             elif var['var_id'] in virtual_ids:
    #                 virtual_true_label.append(var['true_label'])
    #                 virtual_pre_label.append(var['label'])
    #                 virtual_cnt += 1
    #                 virtual_pre_ids.append(var['var_id'])
    #             else:
    #                 hards_true_label.append(var['true_label'])
    #                 hards_pred_label.append(var['label'])
    #                 hard_cnt +=1
    #                 hards_pred_ids.append(var['var_id'])
    #         all_true_label = easys_true_label + hards_true_label
    #         # easy_info = pd.DataFrame()
    #         # easy_info['id'] = easys_pred_ids
    #         # easy_info['pre_label'] = easys_pred_label
    #         # easy_info['true_label'] = easys_true_label
    #         # easy_info.to_csv('2022-6-19-easy.csv', index=None)
    #         #
    #         # virtual_info = pd.DataFrame()
    #         # virtual_info['id'] = virtual_pre_ids
    #         # virtual_info['pre_label'] = virtual_pre_label
    #         # virtual_info['true_label'] = virtual_true_label
    #         # virtual_info.to_csv('2022-6-19-virtual.csv', index=None)
    #         #
    #         # hard_info = pd.DataFrame()
    #         # hard_info['id'] = hards_pred_ids
    #         # hard_info['pre_label'] = hards_pred_label
    #         # hard_info['true_label'] = hards_true_label
    #         # hard_info.to_csv('2022-6-19-hard.csv', index=None)
    #
    #         all_pred_label = easys_pred_label + hards_pred_label
    #         sys.stdout = Logger(self.result)
    #         print("--------------------------------------------")
    #         print("total:")
    #         print("--------------------------------------------")
    #         print("total accuracy_score: " + str(metrics.accuracy_score(all_true_label, all_pred_label)))
    #         print("total precision_score: " + str(metrics.precision_score(all_true_label, all_pred_label)))
    #         print("total recall_score: " + str(metrics.recall_score(all_true_label, all_pred_label)))
    #         print("total f1_score: " + str(metrics.f1_score(all_true_label, all_pred_label)))
    #         print("--------------------------------------------")
    #         print("easys:")
    #         print("--------------------------------------------")
    #         print("easys accuracy_score:" + str(metrics.accuracy_score(easys_true_label, easys_pred_label)))
    #         print("easys precision_score:" + str(metrics.precision_score(easys_true_label, easys_pred_label)))
    #         print("easys recall_score:" + str(metrics.recall_score(easys_true_label, easys_pred_label)))
    #         print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label)))
    #         print("--------------------------------------------")
    #         print("hards:")
    #         print("--------------------------------------------")
    #         print("hards accuracy_score:" + str(metrics.accuracy_score(hards_true_label, hards_pred_label)))
    #         print("hards precision_score: " + str(metrics.precision_score(hards_true_label, hards_pred_label)))
    #         print("hards recall_score: " + str(metrics.recall_score(hards_true_label, hards_pred_label)))
    #         print("hards f1_score: " + str(metrics.f1_score(hards_true_label, hards_pred_label)))
    #         print("--------------------------------------------")
    #         print("virtual:")
    #         print("--------------------------------------------")
    #         print("virtual accuracy_score:" + str(metrics.accuracy_score(virtual_true_label, virtual_pre_label)))
    #         print("virtual precision_score: " + str(metrics.precision_score(virtual_true_label, virtual_pre_label)))
    #         print("virtual recall_score: " + str(metrics.recall_score(virtual_true_label, virtual_pre_label)))
    #         print("virtual f1_score: " + str(metrics.f1_score(virtual_true_label, virtual_pre_label)))
    #         print("easy_cnt is: "+str(easy_cnt))
    #         print("hard_cnt is: " + str(hard_cnt))
    #         print("virtual_cnt is: " + str(virtual_cnt))


    # def score(self):
    #             '''
    #             Get results, including: accuracy, precision, recall, F1 score
    #             :return:
    #             '''
    #             easys_pred_label = list()
    #             easys_true_label = list()
    #             hards_pred_label = list()
    #             hards_true_label = list()
    #             virtual_true_label = list()
    #             virtual_pre_label = list()
    #             easys_pred_ids = list()
    #             virtual_pre_ids = list()
    #             hards_pred_ids = list()
    #             # virtual_ids = list(range(255, 5097)) # TWITTER2013
    #             virtual_ids = list(range(114, 2262))  # cr
    #             # virtual_ids = list(range(426, 8527)) # mr
    #             hard_cnt = 0
    #             easy_cnt = 0
    #             virtual_cnt = 0
    #             for var in self.variables:
    #                 if var['is_easy'] == True:
    #                     easys_true_label.append(var['true_label'])
    #                     easys_pred_label.append(var['label'])
    #                     easy_cnt += 1
    #                     easys_pred_ids.append(var['var_id'])
    #                 elif var['var_id'] in virtual_ids:
    #                     virtual_true_label.append(var['true_label'])
    #                     virtual_pre_label.append(var['label'])
    #                     virtual_cnt += 1
    #                     virtual_pre_ids.append(var['var_id'])
    #                 else:
    #                     hards_true_label.append(var['true_label'])
    #                     hards_pred_label.append(var['label'])
    #                     hard_cnt +=1
    #                     hards_pred_ids.append(var['var_id'])
    #             all_true_label = easys_true_label + hards_true_label
    #             easy_info = pd.DataFrame()
    #             easy_info['id'] = easys_pred_ids
    #             easy_info['pre_label'] = easys_pred_label
    #             easy_info['true_label'] = easys_true_label
    #             # easy_info.to_csv('2022-6-19-easy.csv', index=None)
    #
    #             virtual_info = pd.DataFrame()
    #             virtual_info['id'] = virtual_pre_ids
    #             virtual_info['pre_label'] = virtual_pre_label
    #             virtual_info['true_label'] = virtual_true_label
    #             # virtual_info.to_csv('2022-6-19-virtual.csv', index=None)
    #
    #             hard_info = pd.DataFrame()
    #             hard_info['id'] = hards_pred_ids
    #             hard_info['pre_label'] = hards_pred_label
    #             hard_info['true_label'] = hards_true_label
    #             # hard_info.to_csv('2022-6-19-hard.csv', index=None)
    #
    #             all_pred_label = easys_pred_label + hards_pred_label
    #             sys.stdout = Logger(self.result)
    #             print("--------------------------------------------")
    #             print("total:")
    #             print("--------------------------------------------")
    #             print("total accuracy_score: " + str(metrics.accuracy_score(all_true_label, all_pred_label)))
    #             print("total precision_score: " + str(metrics.precision_score(all_true_label, all_pred_label)))
    #             print("total recall_score: " + str(metrics.recall_score(all_true_label, all_pred_label)))
    #             print("total f1_score: " + str(metrics.f1_score(all_true_label, all_pred_label)))
    #             print("--------------------------------------------")
    #             print("easys:")
    #             print("--------------------------------------------")
    #             print("easys accuracy_score:" + str(metrics.accuracy_score(easys_true_label, easys_pred_label)))
    #             print("easys precision_score:" + str(metrics.precision_score(easys_true_label, easys_pred_label)))
    #             print("easys recall_score:" + str(metrics.recall_score(easys_true_label, easys_pred_label)))
    #             print("easys f1_score: " + str(metrics.f1_score(easys_true_label, easys_pred_label)))
    #             print("--------------------------------------------")
    #             print("hards:")
    #             print("--------------------------------------------")
    #             print("hards accuracy_score:" + str(metrics.accuracy_score(hards_true_label, hards_pred_label)))
    #             print("hards precision_score: " + str(metrics.precision_score(hards_true_label, hards_pred_label)))
    #             print("hards recall_score: " + str(metrics.recall_score(hards_true_label, hards_pred_label)))
    #             print("hards f1_score: " + str(metrics.f1_score(hards_true_label, hards_pred_label)))
    #             print("--------------------------------------------")
    #             print("virtual:")
    #             print("--------------------------------------------")
    #             print("virtual accuracy_score:" + str(metrics.accuracy_score(virtual_true_label, virtual_pre_label)))
    #             print("virtual precision_score: " + str(metrics.precision_score(virtual_true_label, virtual_pre_label)))
    #             print("virtual recall_score: " + str(metrics.recall_score(virtual_true_label, virtual_pre_label)))
    #             print("virtual f1_score: " + str(metrics.f1_score(virtual_true_label, virtual_pre_label)))
    #             print("easy_cnt is: "+str(easy_cnt))
    #             print("hard_cnt is: " + str(hard_cnt))
    #             print("virtual_cnt is: " + str(virtual_cnt))
