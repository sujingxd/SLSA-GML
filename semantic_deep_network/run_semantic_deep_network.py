# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys

TRANSFORMERS_DIR = os.path.abspath(os.path.join(os.getcwd()))
SRC_DIR = os.path.join(TRANSFORMERS_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, TRANSFORMERS_DIR)


import torch
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric,load_from_disk
import torch.nn.functional as F
import pandas as pd

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    glue_output_modes,
    glue_tasks_num_labels,
    GlueDataset,
    XLNetForSequenceClassification,
    RobertaForSequenceClassification,
    AutoModelWithLMHead,
    glue_compute_metrics, BertForSequenceClassification)
from transformers.trainer_utils import  is_main_process
from transformers import GlueDataTrainingArguments

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


def load_labels(data_file_name):
    data_src = pd.read_csv(data_file_name, sep='\t', encoding='utf-8')
    return list(data_src['label'])


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


# def my_run(train_data_set_index, model_name_or_path='roberta-base',task_name='sst-2',do_train=True,do_eval=True,do_predict=True,max_seq_length=128,
#            per_gpu_eval_batch_size=16, per_gpu_train_batch_size=16,learning_rate=2e-5,data_dir='./input_data/CR/',num_train_epochs=3,
#            output_dir='./input_data/CR/', per_device_train_batch_size=16,seed=42,overwrite_output_dir=True,overwrite_cache=True):
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, GlueDataTrainingArguments, TrainingArguments))
    # my_args = {'model_name_or_path': model_name_or_path, 'task_name': task_name, 'do_train': do_train,
    #            'do_eval': do_eval,
    #            'do_predict': do_predict, 'max_seq_length': max_seq_length,
    #            'per_gpu_eval_batch_size': per_gpu_eval_batch_size,
    #            'per_gpu_train_batch_size': per_gpu_train_batch_size, 'learning_rate': learning_rate,
    #            'data_dir': data_dir,
    #            'num_train_epochs': num_train_epochs, 'output_dir': output_dir,
    #            'per_device_train_batch_size': per_device_train_batch_size,
    #            'seed': seed, 'overwrite_output_dir': overwrite_output_dir, 'overwrite_cache': overwrite_cache
    #            }
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    # model_args, data_args, training_args = parser.parse_dict(my_args)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    last_checkpoint = None

    # if (
    #     os.path.exists(training_args.output_dir)
    #     and os.listdir(training_args.output_dir)
    #     and training_args.do_train
    #     and not training_args.overwrite_output_dir
    # ):
    #     raise ValueError(
    #         f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #         "Use --overwrite_output_dir to overcome."
    #     )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
       handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Log on each process the small summary:
  #  logger.warning(
    #    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    #    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
  #  )
    # Set the verbosity to info of the Transformers logger (on main process only):
    #if is_main_process(training_args.local_rank):
     #   transformers.utils.logging.set_verbosity_info()
      #  transformers.utils.logging.enable_default_handler()
       # transformers.utils.logging.enable_explicit_format()
    #logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    #set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    #if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
       # datasets = load_dataset("glue", data_args.task_name)
    #else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
    #    data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
     #   if training_args.do_predict:
        #    if data_args.test_file is not None:
        #        train_extension = data_args.train_file.split(".")[-1]
        #        test_extension = data_args.test_file.split(".")[-1]
       #         assert (
      #              test_extension == train_extension
      #          ), "`test_file` should have the same extension (csv or json) as `train_file`."
      #          data_files["test"] = data_args.test_file
     #       else:
      #          raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

     #   for key in data_files.keys():
      #      logger.info(f"load a local file for {key}: {data_files[key]}")

    #    if data_args.train_file.endswith(".csv"):
      #      # Loading a dataset from local csv files
    #       datasets = load_dataset("csv", data_files=data_files)
    #    else:
            # Loading a dataset from local json files
      #      datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
   # if data_args.task_name is not None:
  #      is_regression = data_args.task_name == "stsb"
   #     if not is_regression:
    #        label_list = datasets["train"].features["label"].names
    #        num_labels = len(label_list)
    #    else:
   #         num_labels = 1
 #   else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
   #     is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
    #    if is_regression:
    #        num_labels = 1
   #     else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
    #        label_list = datasets["train"].unique("label")
      #      label_list.sort()  # Let's sort it for determinism
      #      num_labels = len(label_list)

    is_regression = False
    try:
        num_labels = glue_tasks_num_labels[data_args.task_name]
        output_mode = glue_output_modes[data_args.task_name]
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    train_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, cache_dir=model_args.cache_dir) if training_args.do_train else None # glue数据部分结束
    )
    eval_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="dev", cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    test_dataset = (
        GlueDataset(data_args, tokenizer=tokenizer, mode="test", cache_dir=model_args.cache_dir)
        if training_args.do_predict
        else None
    )

    # Preprocessing the datasets
  #  if data_args.task_name is not None:
  #      sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
 #   else:
  #      # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
     #   non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    #    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
     #       sentence1_key, sentence2_key = "sentence1", "sentence2"
     #   else:
    #        if len(non_label_column_names) >= 2:
    #            sentence1_key, sentence2_key = non_label_column_names[:2]
    #        else:
    #           sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
 #   if data_args.pad_to_max_length:
 #       padding = "max_length"
  #  else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
   #     padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
  #  label_to_id = None
  #  if (
  #      model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
   #     and data_args.task_name is not None
   #     and is_regression
  #  ):
        # Some have all caps in their config, some don't.
   #     label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
   #     if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
   #         label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
   #     else:
   #         logger.warn(
    #            "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #            f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #            "\nIgnoring the model labels as a result.",
    #        )
  #  elif data_args.task_name is None and not is_regression:
    #    label_to_id = {v: i for i, v in enumerate(label_list)}

   # def preprocess_function(examples):
        # Tokenize the texts
   #     args = (
    #        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    #    )
   #     result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
     #   if label_to_id is not None and "label" in examples:
     #       result["label"] = [label_to_id[l] for l in examples["label"]]
     #   return result

   # datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

  #  train_dataset = datasets["train"]
 #   eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
 #   if data_args.task_name is not None or data_args.test_file is not None:
   #     test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
  #  for index in random.sample(range(len(train_dataset)), 3):
   #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue","sst2")
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    data_collator = default_data_collator
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
        num_train_train_examples = len(train_dataset)
        num_written_lines = 0
        temp = trainer.train_data_predict()
        train_predict_out = temp.predictions
        _ids = temp.sid
        train_acc = temp.metrics['eval_accuracy']
        train_labels = temp.label_ids
        logger.info("train_acc is: {}".format(train_acc))
        temp_predictions = torch.from_numpy(train_predict_out)
        temp_predictions = F.softmax(temp_predictions, 1)
        output_train_file = training_args.output_dir + "train_prediction.csv"
        output_train_file0 = training_args.output_dir + "logits_train_prediction.csv"
        np.save(output_train_file0, train_predict_out)
        _pos_prob_list = list()
        _id_list = list()
        _label_list = list()
        for (neg_prob, pos_prob) in temp_predictions:
            pos_prob = pos_prob.item()
            _pos_prob_list.append(pos_prob)
            _id = _ids[num_written_lines]
            _id_list.append(_id)
            _label = train_labels[num_written_lines]
            _label_list.append(_label)
            num_written_lines += 1
        assert num_written_lines == num_train_train_examples
        df = pd.DataFrame()
        df['id'] = _id_list
        df["label"] = _label_list
        df["prob"] = _pos_prob_list
        df = df.sort_values(by="id", ascending=True)
        df.to_csv(output_train_file, index=None)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            eval_result, eval_result_, _ids, _labels = trainer.evaluate(eval_dataset=eval_dataset)
            dev_acc = eval_result['eval_accuracy']
            output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info(f"***** Eval results {task} *****")
                    for key, value in sorted(eval_result.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")

            eval_results.update(eval_result)
            logger.info("dev_acc is: {}".format(dev_acc))
            eval_predictions = F.softmax(torch.from_numpy(eval_result_), dim=1)
            output_dev_file = training_args.output_dir + "dev_prediction.csv"
            output_train_file2 = training_args.output_dir + "logits_dev_prediction.csv"
            np.save(output_train_file2, eval_result)
            num_train_train_examples = len(eval_dataset)
            num_written_lines = 0
            _pos_prob_list = list()
            _id_list = list()
            _label_list = list()
            for (neg_prob, pos_prob) in eval_predictions:
                pos_prob = pos_prob.item()
                _pos_prob_list.append(pos_prob)
                _id = _ids[num_written_lines]
                _id_list.append(_id)
                _label = _labels[num_written_lines]
                _label_list.append(_label)
                num_written_lines += 1
        assert num_written_lines == num_train_train_examples
        df = pd.DataFrame()
        df['id'] = _id_list
        df["label"] = _label_list
        df["prob"] = _pos_prob_list
        df.sort_values(by="id", ascending=True)
        df.to_csv(output_dev_file, index=None)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            #test_dataset.remove_columns_("label")
            test_temp = trainer.predict(test_dataset=test_dataset)
            predictions = test_temp.predictions  # predict
            _ids = test_temp.sid
            num_train_train_examples = len(test_dataset)
            num_written_lines = 0
            temp_predictions = predictions
            dnn_test_file_path = data_args.data_dir + r"test.csv"
            _label_list = list(load_labels(dnn_test_file_path))
            temp_predictions = torch.from_numpy(temp_predictions)
            temp_predictions = F.softmax(temp_predictions, dim=1)
            output_file = training_args.output_dir + "test_prediction.csv"
            output_train_file1 = training_args.output_dir + "logits_test_prediction.csv"
            np.save(output_train_file1, predictions)
            _id_list = []
            _pos_prob_list = []
            for (neg_prob, pos_prob) in temp_predictions:
                pos_prob = pos_prob.item()
                _pos_prob_list.append(pos_prob)
                _id = _ids[num_written_lines]
                _id_list.append(_id)
                num_written_lines += 1
        assert num_written_lines == num_train_train_examples
        df = pd.DataFrame()
        df['id'] = _id_list
        df["label"] = _label_list
        df["prob"] = _pos_prob_list
        df.sort_values(by="id", ascending=True)
        df.to_csv(output_file, index=None)
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
        test_acc = glue_compute_metrics(data_args.task_name, np.array(predictions), np.array(_label_list))
        logger.info("test_acc is: {}".format(test_acc["acc"]))

        import sklearn as sk
        precision = sk.metrics.precision_score(np.array(_label_list), np.array(predictions))
        recall = sk.metrics.recall_score(np.array(_label_list), np.array(predictions))
        f1_score = sk.metrics.f1_score(np.array(_label_list), np.array(predictions))
        logger.info("test_precision is: {}".format(precision))
        logger.info("test_recall is: {}".format(recall))
        logger.info("f1 score is: {}".format(f1_score))
        logger.info("test_acc is: {}".format(test_acc["acc"]))
        dataname = data_args.data_dir.split('/')[-2]
        print('dataname is: ' + dataname + "\n")
        print("seed is:" + str(training_args.seed) + "\t" + "test_acc: " + str(
            test_acc["acc"]) + "\t" + 'precision: ' + str(precision) + "\t" + 'recall: ' + str(
            recall) + "\t" + "f1: " + str(f1_score) + "\n")
        output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info(f"***** Test results {task} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = test_dataset.get_labels()[item]
                        writer.write(f"{index}\t{item}\n")
    _pos_prob_list = [1 if each_pre_prob>= 0.5  else 0 for each_pre_prob in _pos_prob_list]
    # test_index_2_pre_label = dict()
    # for index, each_pre_label in enumerate(_pos_prob_list):
    #     test_index_2_pre_label[test_data_set_index[index]] = each_pre_label
    # return test_index_2_pre_label


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


# def run_dnn_model(train_data_set_index):
#     my_model = my_run(train_data_set_index)
#     return my_model
if __name__ == "__main__":
    main()