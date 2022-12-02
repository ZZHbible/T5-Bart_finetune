#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/11/29
# project = finetune_T5

import pandas as pd
from torch import cuda
from utils import T5Trainer

device = 'cuda' if cuda.is_available() else 'cpu'

# let's define model parameters specific to T5
model_params = {
    "MODEL": "/path/to/t5-large",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 2,  # training batch size
    "VALID_BATCH_SIZE": 2,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
    "SEED": 42,  # set seed for reproducibility
}

path = "./news_summary.csv"
df = pd.read_csv(path)

# T5 accepts prefix of the task to be performed:
# Since we are summarizing, let's add summarize to source text as a prefix
df["text"] = "summarize: " + df["text"]
print(model_params["MODEL"])
T5Trainer(
    dataframe=df,
    source_text="text",
    target_text="headlines",
    model_params=model_params,
    device=device,
    output_dir="outputs_t5",
)
