import pandas as pd
import os


def load_prop(data_file_name, item_name):
    prop_list = list()
    fr = open(data_file_name, encoding='utf-8')
    item_idx = -1
    first_line = True
    for ind, line in enumerate(fr):
        items = [i.strip() for i in line.split('\t')]
        if first_line:
            item_idx = items.index(item_name)
            first_line = False
            continue
        prop_list.append(items[item_idx])
    return prop_list


def load_ids(data_file_name):
    return load_prop(data_file_name, 'id')


def load_labels(data_file_name):
    return load_prop(data_file_name, 'label')


def load_pre_label(data_file_name):
    data_src = pd.read_csv(data_file_name,  encoding='utf-8',sep='\t')
    pre_prob = data_src['prob']
    pre_label = [1 if each_prob >= 0.5 else 0 for each_prob in pre_prob]
    return pre_label


def get_id_and_labels(data_type, file_name_first_part):
    # data_type = r'5%-CR-special\5%-CR-修正unlabeled'
    DIR = r'input_data'
    # tRain_labels = load_labels(os.path.join(file_name_first_part + R"\tRain.csv"))
    # tRain_labels = load_labels(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\tRain.csv')
    # tRain_labels = load_labels(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-{}\tRain.csv'.format(type))
    # tRain_ids = load_ids(os.path.join(file_name_first_part + R"\tRain.csv"))
    # tRain_ids = load_ids(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\tRain.csv')
    # tRain_ids = load_ids(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-{}\tRain.csv'.format(type))
    # tRain_ids = load_ids(os.path.join(file_name_first_part + R"\tRain.csv"))
    # dev_labels = load_labels(os.path.join(file_name_first_part + R"\dev.csv"))
    # dev_ids = load_ids(os.path.join(file_name_first_part + R"\dev.csv"))

    train_labels = load_labels(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\train.csv'.format(DIR,data_type))
    train_ids = load_ids(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\train.csv'.format(DIR,data_type))

    # dev_labels = load_labels(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\dev.csv')
    # dev_ids = load_ids(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\dev.csv')


    dev_labels = load_labels(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\dev.csv'.format(DIR,data_type))
    dev_ids = load_ids(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\dev.csv'.format(DIR,data_type))

    # test_labels = load_labels(os.path.join(file_name_first_part + R"\test.csv"))
    # test_labels = load_labels(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\test.csv')
    test_labels = load_labels(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\test.csv'.format(DIR,data_type))
    # test_ids = load_ids(os.path.join(file_name_first_part + R"\test.csv"))
    # test_ids = load_ids(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\test.csv')
    test_ids = load_ids(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\test.csv'.format(DIR,data_type))
    # tRain_pRe_label = load_pRe_label(os.path.join(file_name_first_part + R"\beRt-base-uncased\tRain_pRediction.csv"))
    # dev_pRe_label = load_pRe_label(os.path.join(file_name_first_part + R"\beRt-base-uncased\dev_pRediction.csv"))
    # test_pRe_label = load_pRe_label(os.path.join(file_name_first_part + R"\beRt-base-uncased\test_pRediction.csv"))

    # tRain_pRe_label = load_pRe_label(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\tRain_pRediction.csv')
    # dev_pRe_label = load_pRe_label(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\dev_pRediction.csv')
    # test_pRe_label = load_pRe_label(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-Right-now\test_pRediction.csv')


    train_pre_label = load_pre_label(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\train_prediction.csv'.format(DIR,data_type))
    # dev_pre_label = load_pre_label(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\dev_prediction.csv'.format(DIR,data_type))
    test_pre_label = load_pre_label(R'E:\服务器上下载的\zcl\transformers-4.3.3\{}\{}\test_prediction.csv'.format(DIR,data_type))


    # test_pRe_label = load_pRe_label(R'E:\服务器上下载的\zcl\tRansfoRmeRs-4.3.3\{}-10\5%-twitter2013-special\test_pRediction.csv')
    pd.DataFrame(train_labels).to_csv(os.path.join(file_name_first_part, 'labels_train.csv')\
                                                  , header=None, index=None)
    # pd.DataFrame(dev_labels).to_csv(os.path.join(file_name_first_part, 'labels_dev.csv')\
    #                                               , header=None, index=None)

    pd.DataFrame(test_labels).to_csv(os.path.join(file_name_first_part, 'labels_test.csv')\
                                                  , header=None, index=None)
    pd.DataFrame(train_ids).to_csv(os.path.join(file_name_first_part, 'paths_train.csv')\
                                                  , header=None, index=None)
    # pd.DataFrame(dev_ids).to_csv(os.path.join(file_name_first_part, 'paths_dev.csv')\
    #                                               , header=None, index=None)

    pd.DataFrame(test_ids).to_csv(os.path.join(file_name_first_part, 'paths_test.csv')\
                                                  , header=None, index=None)

    pd.DataFrame(train_pre_label).to_csv(os.path.join(file_name_first_part, 'predictions_train.csv')\
                                                  , header=None, index=None)
    # pd.DataFrame(dev_pre_label).to_csv(os.path.join(file_name_first_part, 'predictions_dev.csv')\
    #                                               , header=None, index=None)

    pd.DataFrame(test_pre_label).to_csv(os.path.join(file_name_first_part, 'predictions_test.csv')\
                                                  , header=None, index=None)


