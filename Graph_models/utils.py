import torch
import numpy as np


EPS = 1e-8


def rse(preds, labels):
    denom = torch.sum((labels - torch.mean(labels)) ** 2) + EPS
    return torch.sum((preds - labels) ** 2) / denom


def mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))


def mse(preds, labels):
    return torch.mean((preds - labels) ** 2)


def rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))


def wape(preds, labels):
    return torch.sum(torch.abs(preds - labels)) / (torch.sum(torch.abs(labels)) + EPS)


def smape(preds, labels):
    return torch.mean(2.0 * torch.abs(preds - labels) / (torch.abs(preds) + torch.abs(labels) + EPS)) * 100.0


def mape(preds, labels):
    mask = torch.abs(labels) > 1e-4
    if torch.any(mask):
        return torch.mean(torch.abs((preds[mask] - labels[mask]) / labels[mask]))
    return torch.tensor(0.0, device=preds.device, dtype=preds.dtype)


def calc_metrics(preds, labels):
    return rse(preds, labels), mae(preds, labels), mse(preds, labels), mape(preds, labels), rmse(preds, labels), wape(preds, labels), smape(preds, labels)
