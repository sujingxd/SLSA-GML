import numpy as np
import pandas as pd
import  random
import re
path_base = "datasets/semeval14/"
from tqdm import tqdm
from sklearn.utils import shuffle

sentiments=[]
with open('lexicon/SentiWords.txt') as f :
        for line in f.read().strip().splitlines():
            w, s = line.split()
            if float(s) ==0 :
                continue
            sentiments.append(w)
f.close()
import csv


def generate_mr_training_csv(domain):
    n =3
    save_file_train = '{}_train_results'.format(domain)
    save_file_dev = '{}_dev_results'.format(domain)

    df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
    df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')

    # df = pd.concat([df_train, df_dev])
    df = df_train
    df.reset_index(inplace=True, drop=True)
    data_2=[]
    data_3 = []
    print(len(df))
    pos, neg = 0, 0
    df_by_label0 = df[df.label == 0]
    df_by_label1 = df[df.label == 1]

    for i in tqdm(range(len(df))):
        id, src_label, text= df.loc[i][['id','label','text']].values
        src_label = int(src_label)
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = text.replace('\t', " ")
        positive = df_by_label0.sample(n=n).values.tolist()
        negative = df_by_label1.sample(n=n).values.tolist()
        pos+=len(positive)
        neg+=len(negative)
        for c in positive+negative:
            _, trg_label, trg_text = c
            trg_text = trg_text.strip()
            label = 1 if int(trg_label) == int(src_label) else 0
            trg_text = re.sub(' +', ' ', trg_text)
            trg_text = trg_text.replace("\t", " ")
            data_2.append([src_label, text, trg_label, trg_text, label])
            data_3.append([text, trg_text, label])
    print(len(data_2))
    data_train = data_2[:int(0.8*len(data_2))]
    data_dev = data_2[int(0.8*len(data_2)):]
    data_train_use = data_3[:int(0.8*len(data_3))]
    data_dev_use = data_3[int(0.8*len(data_3)):]
    # data_train = data_2

    pd.DataFrame(data_train, columns=['src_label','sentence1','trg_label', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_2_labels_train_train.csv'.format(domain, save_file_train), index=None, sep='\t')
    pd.DataFrame(data_dev, columns=['src_label','sentence1','trg_label', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_2_labels_train_train.csv'.format(domain, save_file_dev), index=None, sep='\t')
    pd.DataFrame(data_train_use, columns=[ 'sentence1', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_train_train.csv'.format(domain, save_file_train), index=None, sep='\t')
    pd.DataFrame(data_dev_use, columns=[ 'sentence1', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_train_train.csv'.format(domain, save_file_dev), index=None, sep='\t')
    pd.DataFrame(data_2, columns=['id_src', 'id_trg', 'sentence1', 'sentence2', 'label']).to_csv(
        'datasets/{0}/{0}_test_results_2_ids.csv'.format(domain), index=None, sep='\t')


def my_generate_mr_training_csv(domain):
    n =0
    save_file_train = '{}_train_results'.format(domain)
    save_file_dev = '{}_dev_results'.format(domain)

    df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
    df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')

    # df = pd.concat([df_train, df_dev])
    df = df_train
    df.reset_index(inplace=True, drop=True)
    data_2=[]
    data_3 = []
    print(len(df))
    pos, neg = 0, 0
    df_by_label0 = df[df.label == 0]
    df_by_label1 = df[df.label == 1]

    for i in tqdm(range(len(df))):
        id, src_label, text= df.loc[i][['id','label','text']].values
        src_label = int(src_label)
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = text.replace('\t', " ")
        positive = df_by_label0.sample(n=n).values.tolist()
        negative = df_by_label1.sample(n=n).values.tolist()
        pos+=len(positive)
        neg+=len(negative)
        for c in positive+negative:
            trg_id, trg_label, trg_text = c
            trg_text = trg_text.strip()
            label = 1 if int(trg_label) == int(src_label) else 0
            trg_text = re.sub(' +', ' ', trg_text)
            trg_text = trg_text.replace("\t", " ")
            data_2.append([id, src_label, text, trg_id, trg_label, trg_text, label])
            data_3.append([text, trg_text, label])
    print(len(data_2))
    data_train = data_2[:int(0.8*len(data_2))]
    data_dev = data_2[int(0.8*len(data_2)):]
    data_train_use = data_3[:int(0.8*len(data_3))]
    data_dev_use = data_3[int(0.8*len(data_3)):]
    # data_train = data_2
    data_train = data_2
    pd.DataFrame(data_train, columns=['src_id','src_label','sentence1','trg_id','trg_label', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_2_labels_train_train.csv'.format(domain, save_file_train), index=None, sep='\t')
    pd.DataFrame(data_dev, columns=['src_label','sentence1','trg_label', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_2_labels_train_train.csv'.format(domain, save_file_dev), index=None, sep='\t')
    pd.DataFrame(data_3, columns=[ 'sentence1', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_train_train.csv'.format(domain, save_file_train), index=None, sep='\t')
    pd.DataFrame(data_dev_use, columns=[ 'sentence1', 'sentence2', 'label']).to_csv('datasets/{0}/{1}_train_train.csv'.format(domain, save_file_dev), index=None, sep='\t')
    pd.DataFrame(data_2, columns=['id_src', 'id_trg', 'sentence1', 'sentence2', 'label']).to_csv('datasets/{0}/{0}_test_results_2_ids.csv'.format(domain), index=None, sep='\t')

def generate_mr_test_csv_cp(domain):
    N=3
    df_test = pd.read_csv('datasets/{}/test.csv'.format(domain), encoding='utf-8', sep='\t')
    df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
    df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')
    # df = pd.concat([df_train, df_test])
    df=df_train
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)

    data_2=[]
    data_3=[]
    data_4 = []
    data_5 = []
    for i in tqdm(range(len(df_test))):
        id_src, src_label, text= df_test.loc[i][['id', 'label', 'text']].values
    # for i in tqdm(range(len(df_train))):
    #     id_src, src_label, text = df_train.loc[i][['id', 'label', 'text']].values
        candidate=df.sample(n=N).values.tolist()
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = text.replace("\t", " ")
        sentence1 = text
        for c in candidate:
            id_trg, trg_label, trg_text = c
            sentence2.append(trg_text) if trg_label == src_label else sentence3.append(trg_text)
            trg_text = trg_text.strip()
            trg_text = re.sub(' +', ' ', trg_text)
            trg_text = trg_text.replace('\t',' ')
            data_2.append([id_src, id_trg, text, trg_text, label])
            data_3.append([src_label, text, trg_label, trg_text, label])
            data_4.append([text, trg_text, label])
            data_5.append([text, trg_text, label])
    # pd.DataFrame(data_2, columns=['id_src','id_trg','sentence1', 'sentence2', 'label']).to_csv('datasets/{0}/{0}_test_results_{1}_ids.csv'.format(domain,N),index=None, sep='\t')
    # pd.DataFrame(data_3, columns=[ 'src_label','sentence1','trg_label', 'sentence2', 'label']).to_csv('datasets/{0}/{0}_test_results_{1}_labels.csv'.format(domain,N), index=None, sep='\t')
    pd.DataFrame(data_4, columns=[ 'sentence1', 'sentence2', 'sentence3']).to_csv('datasets/{0}/{0}_test_results_{1}.csv'.format(domain,N), index=None, sep='\t')
    print(len(data_2))


def generate_mr_test_csv_old(domain):
    N=10
    df_test = pd.read_csv('datasets/{}/test.csv'.format(domain), encoding='utf-8', sep='\t')
    df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
    df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')
    # df = pd.concat([df_train, df_test])
    df=df_train
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)

    data_2=[]
    data_3=[]
    data_4 = []
    data_5 = []
    for i in tqdm(range(len(df_test))):
        id_src, src_label, text= df_test.loc[i][['id', 'label', 'text']].values
        candidate=df.sample(n=N).values.tolist()
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = text.replace("\t", " ")
        sentence1 = text
        sentence2 = []
        sentence3 = []
        for c in candidate:
            id_trg, trg_label, trg_text = c
            if int(trg_label) == int(src_label) and len(sentence2)==0 and trg_text !=sentence1 :
                sentence2 .append(trg_text)
            if int(trg_label) != int(src_label) and len(sentence3)==0:
                sentence3.append(trg_text)
            if len(sentence2) ==1 and len(sentence3)==1:
                data_5.append([sentence1, sentence2, sentence3])
                break
    pd.DataFrame(data_5, columns=[ 'sent0', 'sent1', 'hard_neg']).to_csv('datasets/{0}/{0}_test_results.csv'.format(domain), index=None)
    print(len(data_2))



def generate_mr_train_csv_new(domain):
    N=3
    df_test = pd.read_csv('datasets/{}/test.csv'.format(domain), encoding='utf-8', sep='\t')
    df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
    df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')
    # df = pd.concat([df_train, df_test])
    df=df_train
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)

    data_2=[]
    data_3=[]
    data_4 = []
    data_5 = []
    for i in tqdm(range(len(df_train))):
        id_src, src_label, text= df_train.loc[i][['id', 'label', 'text']].values
        pos_candidate=df[df.label==src_label].sample(n=N).values.tolist()
        neg_candidate = df[df.label != src_label].sample(n=N).values.tolist()
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = text.replace("\t", " ")
        text = text.replace("\"", "")
        text = text.replace("\'", "")
        text = text.replace("\"", "")
        sentence1 = text
        sentence2 = []
        sentence3 = []
        for p,n in zip(pos_candidate,neg_candidate):
            id_trg, trg_label, trg_text = p
            id_trg1, trg_label1, trg_text1 = n
            trg_text = trg_text.replace("\"", "")
            trg_text = trg_text.replace("\'", "")
            trg_text1 = trg_text1.replace("\"", "")
            trg_text1 = trg_text1.replace("\'", "")
            sentence2.append(trg_text)
            sentence3.append(trg_text1)
            if len(sentence2) >=N and len(sentence3)>=N:
                for i in list(range(N)):
                    data_5.append([sentence1, sentence2[i], sentence3[i]])
                # data_5.append([sentence1, sentence2[2], sentence3[2]])
                break
    pd.DataFrame(data_5, columns=[ 'sent0', 'sent1', 'hard_neg']).to_csv('datasets/{0}/{0}_train_results.csv'.format(domain), index=None, sep='\t')
    print(len(data_2))

def generate_mr_test_csv_new(domain):
        N = 2
        df_test = pd.read_csv('datasets/{}/test.csv'.format(domain), encoding='utf-8', sep='\t')
        df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
        df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')
        # df = pd.concat([df_train, df_test])
        df = df_train
        df = shuffle(df)
        df.reset_index(inplace=True, drop=True)

        data_2 = []
        data_3 = []
        data_4 = []
        data_5 = []
        for i in tqdm(range(len(df_test))):
            id_src, src_label, text = df_test.loc[i][['id', 'label', 'text']].values
            pos_candidate = df[df.label == src_label].sample(n=N).values.tolist()
            neg_candidate = df[df.label != src_label].sample(n=N).values.tolist()
            text = text.strip()
            text = re.sub(' +', ' ', text)
            text = text.replace("\t", " ")
            sentence1 = text
            sentence2 = []
            sentence3 = []
            for p, n in zip(pos_candidate, neg_candidate):
                id_trg, trg_label, trg_text = p
                id_trg1, trg_label1, trg_text1 = n
                sentence2.append(trg_text)
                sentence3.append(trg_text1)
                if len(sentence2) >= 1 and len(sentence3) >= 1:
                    data_5.append([sentence1, sentence2[0], sentence3[0]])
                    break
        pd.DataFrame(data_5, columns=['sent0', 'sent1', 'hard_neg']).to_csv(
            'datasets/{0}/{0}_test_results.csv'.format(domain), index=None, sep='\t')
        print(len(data_2))


def generate_mr_dev_csv_new(domain):
    N=2
    df_test = pd.read_csv('datasets/{}/test.csv'.format(domain), encoding='utf-8', sep='\t')
    df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
    df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')
    # df = pd.concat([df_train, df_test])
    df=df_dev
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)

    data_2=[]
    data_3=[]
    data_4 = []
    data_5 = []
    for i in tqdm(range(len(df_dev))):
        id_src, src_label, text= df_dev.loc[i][['id', 'label', 'text']].values
        pos_candidate=df[df.label==src_label].sample(n=N).values.tolist()
        neg_candidate = df[df.label != src_label].sample(n=N).values.tolist()
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = text.replace("\t", " ")
        sentence1 = text
        sentence2 = []
        sentence3 = []
        for p,n in zip(pos_candidate,neg_candidate):
            id_trg, trg_label, trg_text = p
            id_trg1, trg_label1, trg_text1 = n
            sentence2 .append(trg_text)
            sentence3.append(trg_text1)
            if len(sentence2) >=N and len(sentence3)>=N:
                for i in range(N):
                    data_5.append([sentence1, sentence2[i], sentence3[i]])
                break
    pd.DataFrame(data_5, columns=[ 'sent0', 'sent1', 'hard_neg']).to_csv('datasets/{0}/{0}_dev_results.csv'.format(domain), index=None, sep='\t')
    print(len(data_2))

def generate_added_test_csv(domain):
    n = 6
    save_file_train = '{}_train_results'.format(domain)
    save_file_dev = '{}_dev_results'.format(domain)

    df_train = pd.read_csv('datasets/{}/train.csv'.format(domain), encoding='utf-8', sep='\t')
    df_dev = pd.read_csv('datasets/{}/dev.csv'.format(domain), encoding='utf-8', sep='\t')
    df_test = pd.read_csv('datasets/{}/test.csv'.format(domain), encoding='utf-8', sep='\t')

    # df = pd.concat([df_train, df_dev, df_test])
    # df.reset_index(inplace=True, drop=True)
    df_data = df_test
    df = pd.concat([df_train, df_dev])

    data_2 = []
    data_3 = []
    print(len(df))
    pos, neg = 0, 0
    df_by_label1 = df_data[df_data.label == 1]
    df_by_label0 = df_data[df_data.label == 0]

    for i in tqdm(range(len(df))):
        id, src_label, text = df.loc[i][['id', 'label', 'text']].values
        src_label = int(src_label)
        text = text.strip()
        text = re.sub(' +', ' ', text)
        text = text.replace('\t', " ")
        negative = df_by_label0.sample(n=n).values.tolist()
        positive = df_by_label1.sample(n=n).values.tolist()
        pos += len(positive)
        neg += len(negative)
        for c in positive + negative:
            _, trg_label, trg_text = c
            trg_text = trg_text.strip()
            label = 1 if int(trg_label) == int(src_label) else 0
            trg_text = re.sub(' +', ' ', trg_text)
            trg_text = trg_text.replace("\t", " ")
            data_2.append([src_label, text, trg_label, trg_text, label])
            data_3.append([text, trg_text, label])
    print(len(data_2))
    data_train = data_2[:int(0.8 * len(data_2))]
    data_dev = data_2[int(0.8 * len(data_2)):]
    data_train_use = data_3[:int(0.8 * len(data_3))]
    data_dev_use = data_3[int(0.8 * len(data_3)):]
    pd.DataFrame(data_train, columns=['src_label', 'sentence1', 'trg_label', 'sentence2', 'label']).to_csv(
        'datasets/{0}/{1}_2_labels.csv'.format(domain, save_file_train), index=None, sep='\t')
    pd.DataFrame(data_dev, columns=['src_label', 'sentence1', 'trg_label', 'sentence2', 'label']).to_csv(
        'datasets/{0}/{1}_2_labels.csv'.format(domain, save_file_dev), index=None, sep='\t')
    pd.DataFrame(data_train_use, columns=['sentence1', 'sentence2', 'label']).to_csv(
        'datasets/{0}/{1}.csv'.format(domain, save_file_train), index=None, sep='\t')
    pd.DataFrame(data_dev_use, columns=['sentence1', 'sentence2', 'label']).to_csv(
        'datasets/{0}/{1}.csv'.format(domain, save_file_dev), index=None, sep='\t')


if __name__ == '__main__':
    domain='CR'
    # generate_mr_training_csv(domain=domain)
    generate_mr_train_csv_new(domain=domain)
    generate_mr_dev_csv_new(domain=domain)
    generate_mr_test_csv_new(domain=domain)
    # my_generate_mr_training_csv(domain=domain)





