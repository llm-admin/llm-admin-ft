from pprint import pprint
import ray

ray.init(address="auto")

use_gpu = False  # set this to False to run on CPUs
num_workers = 1  # set this to number of GPUs/CPUs you want to use
task = "cola"
model_checkpoint = "distilbert-base-uncased"
batch_size = 16

from datasets import load_dataset

actual_task = "mnli" if task == "mnli-mm" else task
datasets = load_dataset("glue", actual_task)

from datasets import load_metric

def load_metric_fn():
    return load_metric('glue', actual_task)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

import ray.data

ray_datasets = ray.data.from_huggingface(datasets)

import pandas as pd
from ray.data.preprocessors import BatchMapper

def preprocess_function(examples: pd.DataFrame):
    # if we only have one column, we are inferring.
    # no need to tokenize in that case. 
    if len(examples.columns) == 1:
        return examples
    examples = examples.to_dict("list")
    sentence1_key, sentence2_key = task_to_keys[task]
    if sentence2_key is None:
        ret = tokenizer(examples[sentence1_key], truncation=True)
    else:
        ret = tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
    # Add back the original columns
    ret = {**examples, **ret}
    return pd.DataFrame.from_dict(ret)

batch_encoder = BatchMapper(preprocess_function, batch_format="pandas")

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch

num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]
validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
name = f"{model_name}-finetuned-{task}"

def trainer_init_per_worker(train_dataset, eval_dataset = None, **config):
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    metric = load_metric_fn()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    args = TrainingArguments(
        name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=config.get("learning_rate", 2e-5),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=config.get("epochs", 2),
        weight_decay=config.get("weight_decay", 0.01),
        push_to_hub=False,
        disable_tqdm=True,  # declutter the output a little
        no_cuda=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    print("Starting training")
    return trainer

from ray.train.huggingface import TransformersTrainer
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback

trainer = TransformersTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
    datasets={
        "train": ray_datasets["train"],
        "evaluation": ray_datasets[validation_key],
    },
    run_config=RunConfig(
        callbacks=[MLflowLoggerCallback(experiment_name=name)],
        checkpoint_config=CheckpointConfig(
            num_to_keep=1,
            checkpoint_score_attribute="eval_loss",
            checkpoint_score_order="min",
        ),
    ),
    preprocessor=batch_encoder,
)

result = trainer.fit()
