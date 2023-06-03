
from bert_serving.client import BertClient
import pandas as pd
import numpy as np
from tqdm import tqdm

# bc = BertClient(check_length=False)
bc = BertClient()
doc_vecs = bc.encode(['First do it', 'then do it right', 'then do it better'])
print(doc_vecs)
def get_sentence_embedding(sentence):
    embedding = bc.encode([sentence])
    return embedding


def save_matrix(arr_result, file_name):
    f_out = open(file_name, 'w')
    np.set_printoptions(threshold=np.inf)
    np.savetxt(f_out, arr_result, fmt='%.17f')
    f_out.close()


def load_mr(file_path):
    mr_csv_data = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    all_lines = mr_csv_data['text']
    all_vector_representation = np.zeros(shape=(len(all_lines), 768))
    for i, line in tqdm(enumerate(all_lines)):
        all_vector_representation[i] = get_sentence_embedding(line)
    save_matrix(all_vector_representation, file_path+".sentence.csv")

#
# if __name__ == "__main__":
#     load_mr('data/mr/mr_train_results.csv')
#     load_mr('data/mr/mr_dev_results.csv')
#     load_mr('data/mr/mr_test_results.csv')
