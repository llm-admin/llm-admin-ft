from ._base import BaseFT
from abc import ABC, abstractmethod
from llmadmin.backend.logger import get_logger
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from typing import Union
from llmadmin.backend.server.models import FTApp
from datasets import load_dataset
from datasets import load_metric
import pandas as pd
from ray.data.preprocessors import BatchMapper
import ray
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
from ray.train.huggingface import TransformersTrainer
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.air.integrations.mlflow import MLflowLoggerCallback
from .utils import parse_task_name 
from .tasks import TASK_REGISTRY
from .tasks._base import Task
from ray.train.huggingface import TransformersCheckpoint
from .const import CHECKPOINT_PATH
from .callback import CustomCallback



logger = get_logger(__name__)


class TransformersFT(BaseFT):
    def __init__(
        self,
        ftApp: FTApp,
    ):
        super().__init__(
            ftapp=ftApp
        )
    
    def train(self):
        taskobj: Task = None
        task = parse_task_name(self.ftapp)
        taskcls = TASK_REGISTRY[task]

        if not taskcls:
            logger.error(
                    f"Couldn't load defined task from register: {task}"
                )
            raise

        tokenizer = self.initializer.load_tokenizer(self.model_config.model_id)
        taskobj = taskcls.from_tokenizer(tokenizer, self.ftapp.ft_config)
        from_pretrained_kwargs = taskobj.FROM_PRETRAINED_KWARGS if taskobj.FROM_PRETRAINED_KWARGS else {}
        model = self.initializer.load_model(self.model_config.model_id, taskobj.AUTO_MODEL_CLASS, **from_pretrained_kwargs)
        taskobj.set_model(model)

        preprocess_function = taskobj.get_data_proprocess()
        compute_metrics_function = taskobj.get_compute_metrics()
        data_collator = taskobj.get_data_collator()
        batch_encoder = BatchMapper(preprocess_function, batch_format="pandas")
        
        ray_datasets = ray.data.from_huggingface(taskobj.get_dataset())
        model_name = self.model_config.model_id.split("/")[-1]
        task = self.ft_task
        name = f"{model_name}-finetuned-{task}"
        use_gpu = True if torch.cuda.is_available() else False

        def trainer_init_per_worker(train_dataset, eval_dataset = None, **config):
            print(f"Is CUDA available: {torch.cuda.is_available()}")

            args = TrainingArguments(
                name,
                evaluation_strategy=config.get("evaluation_strategy", "epoch"),
                save_strategy=config.get("save_strategy", "epoch"),
                logging_strategy=config.get("logging_strategy", "epoch"),
                logging_steps = 2,
                save_steps = 500,
                eval_steps = 2,
                learning_rate=config.get("learning_rate", 2e-5),
                per_device_train_batch_size=config.get("per_device_train_batch_size", 16),
                per_device_eval_batch_size=config.get("per_device_train_batch_size", 16),
                num_train_epochs=config.get("epochs", 2),
                weight_decay=config.get("weight_decay", 0.01),
                push_to_hub=False,
                disable_tqdm=False,  # declutter the output a little
                no_cuda=not use_gpu,  # you need to explicitly set no_cuda if you want CPUs
                remove_unused_columns=config.get("remove_unused_columns", True),
            )

            trainer = Trainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_function,
                data_collator=data_collator,
            )
            trainer.add_callback(CustomCallback(trainer))
            print("Starting training")

            return trainer
        
        trainer = TransformersTrainer(
            trainer_init_per_worker=trainer_init_per_worker,
            trainer_init_config = self.train_conf.get_train_kwargs(),
            scaling_config=self.scale_config.as_air_scaling_config(),
            datasets={
                "train": ray_datasets[taskobj.training_key()],
                "evaluation": ray_datasets[taskobj.validation_key()],
            },
            run_config=RunConfig(
                # callbacks=[MLflowLoggerCallback(experiment_name=name)],
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="eval_loss",
                    checkpoint_score_order="min",
                ),
            ),
            preprocessor=batch_encoder,
        )

        result = trainer.fit()
        print(result)
        checkpoint = TransformersCheckpoint.from_checkpoint(result.checkpoint)
        hf_trainer = checkpoint.get_model(model=taskobj.AUTO_MODEL_CLASS)
        hf_trainer.save_pretrained(CHECKPOINT_PATH)
        tokenizer.save_pretrained(CHECKPOINT_PATH)

        print("Done")




