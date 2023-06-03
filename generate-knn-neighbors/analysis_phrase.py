import json
import pandas as pd
import os
import sys
TRANSFORMERS_DIR = os.path.abspath(os.path.join(os.getcwd()))
SRC_DIR = os.path.join(TRANSFORMERS_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, TRANSFORMERS_DIR)
from stanfordcorenlp import StanfordCoreNLP
# from nltk.parse.stanford import StanfordDependencyParser
# from transformers import BertTokenizer, BertForMaskedLM
import pandas as pd
from nltk import word_tokenize
corenlp_parser =StanfordCoreNLP(r'E:\stanford-corenlp-full-2018-02-27')


NEGATE_WORD = {"aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
 "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
 "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
 "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
 "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
 "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
 "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
 "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite","n't","no","not",'\'t'}

def get_lexicon():
    lexicon = dict()
    lexicon_path = r"E:\RM_SA_Club-master\input_data\lexicon\scores_mr_big.txt"
    with open(lexicon_path, encoding='utf-8') as f:
        for line in f:
            measure = line.strip().split('\t')[1]
            word = line.strip().split('\t')[0]
            if abs(float(measure)) > 0:
                lexicon[word] = float(measure)
    return lexicon

def toTree(expression):
    count = 0
    tree = dict()
    express = dict()
    phrase_dict = dict()
    whole_dict = dict()
    msg = ""
    stack = list()
    for char in expression:
        if char == '(':
            count += 1
            if msg in phrase_dict.keys():
                phrase_dict[msg] += 1
                if phrase_dict[msg] % 2 == 1:
                    stack.append(msg + '_' + str(phrase_dict[msg] / 2))
                else:
                    if phrase_dict[msg] != 2:
                        stack.append(msg + '_' + str((phrase_dict[msg] - 1) / 2))
                    else:
                        stack.append(msg)
            else:
                phrase_dict[msg] = 1
                stack.append(msg)
            msg = ""
        elif char == ')':
            parent = stack.pop()

            if ' ' in msg:
                msg = msg[msg.find(' ') + 1:]
                express[msg] = msg

            if parent not in tree:
                tree[parent] = list()
            # print(msg)
            if msg in whole_dict.keys():
                whole_dict[msg] += 1
                # if " " in msg:
                tree[parent].append(msg + ' ' + str(whole_dict[msg] - 1))
                whole_dict[msg + ' ' + str(whole_dict[msg] - 1)] = 1
            else:
                whole_dict[msg] = 1
                tree[parent].append(msg)

            if parent == '':
                continue
            if parent not in express.keys():
                express[parent] = express[msg]
            else:
                express[parent] += ' ' + express[msg]
            msg = parent
        else:
            msg += char
    return tree, express


def analysis_sentiment_tree(tree_text):
    tree_text = tree_text.replace('\n', '')
    tree_text = tree_text.replace('\r', '')
    tree_text = tree_text.replace('  ', '')
    tree_text = tree_text.replace(' (', '(')
    tree, express = toTree(tree_text)
    phrase_senti_map = dict()
    for exp_key in express.keys():
        exp_val = express[exp_key]
        exp_items = exp_key.split('|')
        if len(exp_items) != 3:
            assert len(exp_items) == 1
            continue
        sentiment = exp_items[1].split('=')
        if len(sentiment) != 2:
            assert 0
            continue
        senti_val = int(sentiment[1])
        if senti_val == 2:
            continue
        polarity = 0 if senti_val < 2 else 1
        if exp_val not in phrase_senti_map:
            phrase_senti_map[exp_val] = polarity
        else:
            phrase_senti_map[exp_val] = str(phrase_senti_map[exp_val]) + ',' + str(polarity)
    return phrase_senti_map

def generate_file(data_dir, file_name):
    lexicon = get_lexicon()
    data_path = data_dir + file_name
    data_info = pd.read_csv(data_path, sep='\t', encoding='utf-8')
    all_new_rows = list()
    all_diff_rows = list()
    for idx, row in data_info.iterrows():
        sid = row['id']
        text = row['text']
        # label = row['label']
        # text = "its positive , so a positive positive can 't easily positive it over ."
        new_text = text
        pos_text = text
        neg_text = text
        words = word_tokenize(text)
        senti_flag = False
        not_flag = False
        for word in words:
            if word in lexicon:
                senti_flag = True
                pos_text = pos_text.replace(word, 'positive') # 新文本
                neg_text = neg_text.replace(word, 'negative') # 新文本
            if word in NEGATE_WORD:
                not_flag = True
                pos_text = pos_text.replace("n't", "")
                pos_text = pos_text.replace("nt", "")
                pos_text = pos_text.replace("n 't", "")
                pos_text = pos_text.replace("n\'t", "")
                pos_text = pos_text.replace("not", "")
                pos_text = pos_text.replace("without", "has")
                pos_text = pos_text.replace("neither", "")
                pos_text = pos_text.replace("never", "")
                pos_text = pos_text.replace("nor", "")
                pos_text = pos_text.replace("no", "")
                pos_text = pos_text.replace("\'t", "")
                neg_text = neg_text.replace("n't", "")
                neg_text = neg_text.replace("nt", "")
                neg_text = neg_text.replace("n 't", "")
                neg_text = neg_text.replace("n\'t", "")
                neg_text = neg_text.replace("not", "")
                neg_text = neg_text.replace("without", "has")
                neg_text = neg_text.replace("neither", "")
                neg_text = neg_text.replace("never", "")
                neg_text = neg_text.replace("nor", "")
                neg_text = neg_text.replace("no", "")
                neg_text = neg_text.replace("\'t", "")
        # if senti_flag == False and not_flag == False:
        #     tag_word_pair = corenlp_parser.pos_tag(new_text.lower())
        #     modify_flag = False
        #     for index, (word, tag) in enumerate(tag_word_pair):
        #         if tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP']:
        #             pos_text = pos_text.replace(word, 'positive')
        #             neg_text = neg_text.replace(word, 'negative')
        #             modify_flag = True
        #     if modify_flag == False:
        #         for index, (word, tag) in enumerate(tag_word_pair):
        #             if tag in ['JJ', 'JJR', 'JJS']:
        #                 pos_text = pos_text.replace(word, 'positive')
        #                 neg_text = neg_text.replace(word, 'negative')
        if pos_text == neg_text:
            continue
        all_new_rows.append([sid, 1, pos_text])
        all_new_rows.append([sid, 0, neg_text])
    new_data_path = data_dir + "new-" + file_name
    pd.DataFrame(all_new_rows).to_csv(new_data_path, header=['id','label', 'new_text'],
                                      sep='\t', index=None)

def get_sentiemnet_tree():
    arg_text_csv_json = 'new-test-CR-test-demo.txt.json'
    with open(arg_text_csv_json, 'r') as f:
        data = json.load(f)
    expression = []
    expression_map = []
    scores = []
    sen_set = data["sentences"]
    for sen in sen_set:
        tree_text = sen["sentimentTree"]
        # expression.append(tree_text)
        # expression_map.append(analysis_sentiment_tree(tree_text))
        score = int(tree_text.split("|")[1].split('=')[1])
        scores.append(score)
    # expression_csv = pd.DataFrame()
    # # expression_csv['sentiment_tree'] = expression
    # expression_csv['sentiment_map'] = expression_map
    # expression_csv.to_csv("sentiment_tree.txt", index=False)
    return scores


def get_sentiment_tree(arg_text_csv_json):
    lex = get_lexicon()
    tctd = pd.read_csv('test-CR-test-demo.csv')
    all_texts = tctd['text']
    for txt in all_texts:
        tokens = txt.split(" ")
        senti_words = list()
        for token in tokens:
            if token in lex:
                senti_words.append(token)
        all_senti_train.append(senti_words)
    dev_data['senti_words'] = all_senti_train
    dev_data.to_csv("./input_data/CR/test_senti.csv", index=None)

    with open(arg_text_csv_json, 'r') as f:
        data = json.load(f)
    expression = []
    expression_map = []
    sen_set = data["sentences"]
    for sen in sen_set:
        tree_text = sen["sentimentTree"]
        expression.append(tree_text)
        expression_map.append(analysis_sentiment_tree(tree_text))
    expression_csv = pd.DataFrame()
    # expression_csv['sentiment_tree'] = expression
    expression_csv['sentiment_map'] = expression_map
    expression_csv.to_csv("sentiment_tree.txt", index=False)
    pos_sentence = list()

    # for index, value in enumerate(expression_map): # generate pos sentence
    #     cnt = 0
    #     for index, (k, v) in enumerate(value):
    #         if v ==1:
    #             cnt += 1
    #     if cnt == len(value):
    #         pos_sentence.append(value)



    return expression


def main():
    # get_sentiment_tree('test-CR-test-demo.csv.json')
    data_dir =r'G:\SentiLARE-master\input_data\CR\\'
    file_name ='test-CR-test-demo.csv'
    generate_file(data_dir, file_name)
    new_data_path = data_dir + "new-" + file_name
    # pre_labels = get_sentiemnet_tree()
    # new_pre_labels = list()
    # prn = pd.read_csv(new_data_path, sep='\t')
    # for i in pre_labels:
    #     if i <=1:
    #         new_pre_labels.append(0)
    #     else:
    #         new_pre_labels.append(1)
    # prn['pre_label'] = new_pre_labels
    # prn.to_csv(data_dir + "new-new-" + file_name,index=None)
def judge():
    pdr = pd.read_csv('CR_test_prediction.csv')
    dec_txt = pdr['text']
    desc = pd.read_csv('test_sentence_pair-1-近邻.csv')
    src_txt = desc['src']
    cnt = 1
    for txt in src_txt:
        for second in dec_txt:
            if txt == second:
                cnt += 1
                break
    print("cnt: "+str(cnt))


if __name__ == '__main__':
    main()
    # judge()




