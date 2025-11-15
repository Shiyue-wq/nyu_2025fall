#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 22:12:40 2025

@author: duilzhang
"""

def compute_accuracy_from_file(path):
    preds, labels = [], []
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        for i in range(0, len(lines), 2):
            preds.append(int(lines[i]))
            labels.append(int(lines[i+1]))
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / len(preds)
    return acc

# 原始模型结果
acc_original = compute_accuracy_from_file("/Users/duilzhang/Downloads/hw4_result/out_original.txt")

# 新模型（例如带 transformation 的）
acc_augmented = compute_accuracy_from_file("/Users/duilzhang/Downloads/hw4_result/out_augmented_original.txt")

print(f"Original Model Accuracy: {acc_original:.5f}")
print(f"Augmented Model Accuracy: {acc_augmented:.5f}")
print(f"Improvement: {(acc_augmented - acc_original) * 100:.2f}%")
