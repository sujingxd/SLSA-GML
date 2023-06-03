import os
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
import paddle



if __name__ == '__main__':
    # tokenizer = SkepTokenizer.from_pretrained(pretrained_model_name_or_path="skep_ernie_2.0_large_en")
    save_path = ''
    model = SkepForSequenceClassification.from_pretrained(pretrained_model_name_or_path='skep_ernie_2.0_large_en', num_classes=2)
    state_dict = paddle.load(save_path)
    model.set_dict(state_dict)
    tokenizer = SkepTokenizer.from_pretrained(save_path)
