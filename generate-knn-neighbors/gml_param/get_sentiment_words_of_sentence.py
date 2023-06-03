import pandas as pd
import pickle


def get_sentiment_words():
    # epa_txt = open(r'E:\2021-xiangrogn-kbs\EnglishWords_EPAs.xlsx','r',encoding='utf-8')
    vad_txt = open(r'E:\2021-xiangrogn-kbs\NRC-VAD-Lexicon.txt','r',encoding='utf-8')
    epa_txt = pd.read_excel(r'E:\2021-xiangrogn-kbs\EnglishWords_EPAs.xlsx')
    vad_dic = dict()
    epa_dic = dict()
    for line in vad_txt:
        arr_line = line.split("\t")
        wd = arr_line[0]
        score = []
        score.append(eval(arr_line[1]))
        score.append(eval(arr_line[2]))
        score.append(eval(arr_line[3].strip()))
        vad_dic[wd] = score
    for arr_line in epa_txt.values:
        wd = arr_line[0]
        score = []
        score.append(arr_line[1])
        score.append(arr_line[2])
        score.append(arr_line[3])
        epa_dic[wd] = score
    # temp0 = vad_dic
    # temp1 = epa_dic
    fw0 = open('vad.pkl','wb')
    fw1 = open('epa.pkl','wb')
    pickle.dump(vad_dic, fw0)
    pickle.dump(epa_dic, fw1)


if __name__ == "__main__":
    get_sentiment_words()