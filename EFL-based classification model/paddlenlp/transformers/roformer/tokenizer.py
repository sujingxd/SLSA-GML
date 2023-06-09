# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import jieba
from ..bert.tokenizer import BasicTokenizer, WordpieceTokenizer
from ..tokenizer_utils import PretrainedTokenizer

__all__ = ["RoFormerTokenizer", "JiebaBasicTokenizer"]


class JiebaBasicTokenizer(BasicTokenizer):
    """
    Runs basic tokenization with jieba (punctuation splitting, lower casing, jieba pretokenizer etc.).
    Args:
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. If you use the RoFormer Pretrained model, lower is set to
            Flase when using the cased model, otherwise it is set to True.
            Default: True.
    """

    def __init__(self, vocab, do_lower_case=True):
        """Constructs a JiebaBasicTokenizer."""
        self.vocab = vocab
        self.do_lower_case = do_lower_case

    def _tokenize_chinese_chars(self, text):
        output = []
        for wholeword in jieba.cut(text, HMM=False):
            if wholeword in self.vocab:
                output.append(" ")
                output.append(wholeword)
                output.append(" ")
            else:
                for char in wholeword:
                    cp = ord(char)
                    if self._is_chinese_char(cp):
                        output.append(" ")
                        output.append(char)
                        output.append(" ")
                    else:
                        output.append(char)
        return "".join(output)


class RoFormerTokenizer(PretrainedTokenizer):
    """
    Constructs a RoFormer tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing, jieba pretokenizer and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    Args:
        vocab_file (str): file path of the vocabulary
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. If you use the RoFormer pretrained model, lower is set to
            Flase when using the cased model, otherwise it is set to True.
            Default: True.
        use_jieba (bool): Whether or not to tokenize the text with jieba. Default: False.
        unk_token (str): The special token for unkown words. Default: "[UNK]".
        sep_token (str): The special token for separator token . Default: "[SEP]".
        pad_token (str): The special token for padding. Default: "[PAD]".
        cls_token (str): The special token for cls. Default: "[CLS]".
        mask_token (str): The special token for mask. Default: "[MASK]".

    Examples:
        .. code-block:: python
            from paddlenlp.transformers.roformer import RoFormerTokenizer
            tokenizer = RoFormerTokenizer.from_pretrained('roformer-chinese-base')
            # the following line get: ['今天', '的', '天气', '非常', '好', '！']
            tokens = tokenizer.tokenize('今天的天气非常好！')
            # the following line get: '今天 的 天气 非常 好 ！'
            tokenizer.convert_tokens_to_string(tokens)
    """

    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            # chinese word level model
            "roformer-chinese-small":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-small/vocab.txt",
            "roformer-chinese-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-base/vocab.txt",
            # chinese char level model
            "roformer-chinese-char-small":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-char-small/vocab.txt",
            "roformer-chinese-char-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-char-base/vocab.txt",
            "roformer-chinese-sim-char-ft-small":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-sim-char-ft-small/vocab.txt",
            "roformer-chinese-sim-char-ft-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-sim-char-ft-base/vocab.txt",
            "roformer-chinese-sim-char-small":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-sim-char-small/vocab.txt",
            "roformer-chinese-sim-char-base":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-chinese-sim-char-base/vocab.txt",
            # english
            "roformer-english-small-discriminator":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-english-small-discriminator/vocab.txt",
            "roformer-english-small-generator":
            "https://paddlenlp.bj.bcebos.com/models/transformers/roformer/roformer-english-small-generator/vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "roformer-chinese-small": {
            "do_lower_case": True,
            "use_jieba": True
        },
        "roformer-chinese-base": {
            "do_lower_case": True,
            "use_jieba": True
        },
        "roformer-chinese-char-small": {
            "do_lower_case": True,
            "use_jieba": False
        },
        "roformer-chinese-char-base": {
            "do_lower_case": True,
            "use_jieba": False
        },
        "roformer-chinese-sim-char-ft-small": {
            "do_lower_case": True,
            "use_jieba": False
        },
        "roformer-chinese-sim-char-ft-base": {
            "do_lower_case": True,
            "use_jieba": False
        },
        "roformer-chinese-sim-char-small": {
            "do_lower_case": True,
            "use_jieba": False
        },
        "roformer-chinese-sim-char-base": {
            "do_lower_case": True,
            "use_jieba": False
        },
        "roformer-english-small-discriminator": {
            "do_lower_case": True,
            "use_jieba": False
        },
        "roformer-english-small-generator": {
            "do_lower_case": True,
            "use_jieba": False
        },
    }
    padding_side = "right"

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            use_jieba=False,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]", ):

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the "
                "vocabulary from a pretrained model please use "
                "`tokenizer = RoFormerTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
                .format(vocab_file))
        self.vocab = self.load_vocabulary(vocab_file, unk_token=unk_token)
        if use_jieba:
            self.basic_tokenizer = JiebaBasicTokenizer(
                vocab=self.vocab, do_lower_case=do_lower_case)
        else:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(
            vocab=self.vocab, unk_token=unk_token)

    @property
    def vocab_size(self):
        """
        return the size of vocabulary.
        Returns:
            int: the size of vocabulary.
        """
        return len(self.vocab)

    def _tokenize(self, text):
        """
        End-to-end tokenization for RoFormer models.
        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of string representing converted tokens.
        """
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def tokenize(self, text):
        """
        End-to-end tokenization for RoFormer models.
        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.
        Args:
            tokens (list): A list of string representing tokens to be converted.
        Returns:
            str: Converted string from tokens.
        """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens.

        A Roformer sequence has the following format:
        ::
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens.

        A RoFormer offset_mapping has the following format:
        ::
            - single sequence: ``(0,0) X (0,0)``
            - pair of sequences: `(0,0) A (0,0) B (0,0)``

        Args:
            offset_mapping_ids_0 (:obj:`List[tuple]`):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (:obj:`List[tuple]`, `optional`):
                Optional second list of char offsets for offset mapping pairs.

        Returns:
            :obj:`List[tuple]`: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.

        A RoFormer sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 +
                                                          _sep) * [1]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optinal): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already
                formatted with special tokens for the model. Defaults to None.

        Returns:
            results (List[int]): The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(
                map(
                    lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0,
                    token_ids_0, ))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
