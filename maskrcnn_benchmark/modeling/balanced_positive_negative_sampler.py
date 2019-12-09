# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from maskrcnn_benchmark.utils.tensor_saver import get_tensor_saver


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction, random_sample):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.random_sample = random_sample

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for img_idx, matched_idxs_per_image in enumerate(matched_idxs):
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)
            # get_tensor_saver().save(
            #     tensor=positive,
            #     tensor_name="pos_indices_{}".format(img_idx),
            #     scope="rpn",
            #     save_grad=False
            # )
            # get_tensor_saver().save(
            #     tensor=negative,
            #     tensor_name="neg_indices_{}".format(img_idx),
            #     scope="rpn",
            #     save_grad=False
            # )

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            if self.random_sample:
                perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
                perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
            else:
                perm1 = torch.arange(positive.numel(), device=positive.device)[:num_pos]
                perm2 = torch.arange(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
