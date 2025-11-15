#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:25:45 2025

@author: duilzhang
"""
from load_data import load_lines
import numpy as np
from transformers import T5TokenizerFast
tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")


train_nl = load_lines('data/train.nl')
dev_nl   = load_lines('data/dev.nl')
train_sql = load_lines('data/train.sql')
dev_sql   = load_lines('data/dev.sql')



def compute_raw_stats(nl_list, sql_list):
    num_examples = len(nl_list)
    num_ex_sql = len(sql_list)
    mean_nl_len = np.mean([len(x.split()) for x in nl_list])
    mean_sql_len = np.mean([len(x.split()) for x in sql_list])
    vocab_nl = set(word for x in nl_list for word in x.split())
    vocab_sql = set(word for x in sql_list for word in x.split())
    return num_examples,num_ex_sql, mean_nl_len, mean_sql_len, len(vocab_nl), len(vocab_sql)



def compute_tokenized_stats(nl_list, sql_list, tokenizer):
    nl_tokens = [tokenizer.encode(x, truncation=True, max_length=512) for x in nl_list]
    sql_tokens = [tokenizer.encode(x, truncation=True, max_length=512) for x in sql_list]
    mean_nl_len = np.mean([len(t) for t in nl_tokens])
    mean_sql_len = np.mean([len(t) for t in sql_tokens])
    vocab_nl = set(token for seq in nl_tokens for token in seq)
    vocab_sql = set(token for seq in sql_tokens for token in seq)
    return mean_nl_len, mean_sql_len, len(vocab_nl), len(vocab_sql)



print("==========RAW DATA==========")
print("Train:", compute_raw_stats(train_nl, train_sql))
print("Dev:", compute_raw_stats(dev_nl, dev_sql))

print("============PREPROCESSING=========")
print("Train:", compute_tokenized_stats(train_nl, train_sql, tokenizer))
print("Dev:", compute_tokenized_stats(dev_nl, dev_sql, tokenizer))