# SLSA-GML
It is an Supervised Gradual Machine Learning approach for Sentence-Level Sentiment Analysis task.
## Download Dataset
You can download dataset by yourself. MR dataset is available from https://www.cs.cornell.edu/
people/pabo/movie-review-data/. CR dataset is available from https://www.cs.uic.edu/~liub/FBS/
sentiment-analysis.html#datasets. Twitter2013 dataset is available from https://www.dropbox.com/s/
byzr8yoda6bua1b/2017_English_final.zip?file_subpath=%2F2017_English_final%2FGOLD%2F\
Subtask_A. SST dataset is available from http://nlp.stanford.edu/sentiment. The sentiment lexicon EPA
used in our paper is available from http://www.indiana.edu/~socpsy/public_files/EnglishWords_EPAs.
xlsx, and another sentiment lexicon VAD is available from https://saifmohammad.com/WebPages/nrc-vad.
html.
## Requirements
transformers==4.3.3
torch==1.7.1
en_core_web_sm=2.1.0
## Data Process
### Generate sentence pairs
generate_training-pairs.py
## The whole pineline

### generate semantic-related facotr
EFL-based classification model/examples/few_shot/efl/train-single.py  
### generate knn-related facotr
transformers-modify/examples/run_glue.py
### run gml
gml-master/example.py
