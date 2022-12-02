#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/11/29
# project = eval

import pandas as pd
import evaluate
df=pd.read_csv("./output_bart/predictions.csv")
rouge=evaluate.load('rouge')
predictions=df['Generated Text']
references=df['Actual Text']
results=rouge.compute(predictions=predictions,references=references)
print(results)