#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/12/2
# project = finetune_Bart.py

import pandas as pd
import yaml
import argparse
from utils import BARTTrainer

# Importing the T5 modules from huggingface/transformers
# Setting up the device for GPU usage
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

# let's define model parameters specific to T5
model_params = {
    "MODEL": "/path/to/bart-large",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 8,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}



path = "/home/user/cpei/zhello/T5_tutorial/news_summary.csv"
df = pd.read_csv(path)


# T5 accepts prefix of the task to be performed:
# Since we are summarizing, let's add summarize to source text as a prefix
df["text"] = "summarize: " + df["text"]
print(model_params["MODEL"])
BARTTrainer(
    dataframe=df,
    source_text="text",
    target_text="headlines",
    model_params=model_params,
    device=device,
    output_dir="outputs",
)

