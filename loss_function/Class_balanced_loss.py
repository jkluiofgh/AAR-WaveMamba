# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:37:49 2021

@author: axmao2-c
"""

"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
   
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""


import numpy as np
import torch
import torch.nn.functional as F



def focal_loss(labels, logits, alpha, gamma):
 
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss



def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
  
    device = logits.device
    labels = labels.to(device)
    
    if not isinstance(samples_per_cls, torch.Tensor):
        samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float)
    samples_per_cls = samples_per_cls.to(device)

    # 检查标签值是否在有效范围内
    if labels.max() >= no_of_classes:
        raise ValueError(
            f"Labels contain invalid class index. Max label: {labels.max()}, "
            f"num_classes: {no_of_classes}, unique labels: {torch.unique(labels)}"
        )
    
    # 计算权重############################################################################
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / torch.sum(weights) * no_of_classes
    weights = weights.to(device)

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)
    ########################################################################################
    
    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss
