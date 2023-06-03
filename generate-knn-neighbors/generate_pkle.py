import pandas as pd
data_file_name = r"E:\服务器上下载的\zcl\transformers-4.3.3\input_data-1\1%-twitter2016-special\生成easy"
data_src = pd.read_csv(data_file_name+r"\test.csv", sep='\t',encoding='gbk')
# data_src = pd.read_csv(data_file_name, sep='\t', encoding='GB2312')
temp =  list(data_src['label'])
aa = temp