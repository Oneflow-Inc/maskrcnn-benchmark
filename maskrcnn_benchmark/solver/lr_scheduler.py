# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from bisect import bisect_right

import torch

APPOINTED_LR = -1.0


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
        appointed_lr=-1.0,
        bias_lr_factor=2,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear", "none"):
            raise ValueError(
                "Only 'constant' or 'linear' or 'none' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        global APPOINTED_LR
        APPOINTED_LR = appointed_lr
        self.bias_lr_factor = bias_lr_factor
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        global APPOINTED_LR
        if APPOINTED_LR >= 0.0:
            lr_cls, lrs = cluster(self.base_lrs)
            assert len(lrs) == 2
            lrs = [APPOINTED_LR]
            lrs += [APPOINTED_LR * self.bias_lr_factor]
            return [lrs[i] for i in lr_cls]

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def cluster(x):
    import numpy as np

    uniq = np.unique(np.array(x))
    ids = []
    for v in x:
        idx = np.where(v == uniq)[0].item()
        ids.append(idx)
    return ids, uniq
