import numpy as np


class BinarySegmentationMeter:
    def __init__(self, threshold: float = 0.5, eps: float = 1e-8):
        self.threshold = threshold
        self.eps = eps
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred, gt) -> None:
        pred = np.asarray(pred) >= self.threshold
        gt = np.asarray(gt) >= self.threshold

        self.tp += int(np.logical_and(pred, gt).sum())
        self.fp += int(np.logical_and(pred, np.logical_not(gt)).sum())
        self.fn += int(np.logical_and(np.logical_not(pred), gt).sum())
        self.tn += int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())

    def show(self):
        precision = self.tp / (self.tp + self.fp + self.eps)
        recall = self.tp / (self.tp + self.fn + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + self.eps)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "F1": f1,
        }
