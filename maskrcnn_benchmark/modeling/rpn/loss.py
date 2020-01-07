# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

from maskrcnn_benchmark.utils.tensor_saver import get_tensor_saver


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, img_idx, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        get_tensor_saver().save(
            tensor=match_quality_matrix,
            tensor_name="CHECK_POINT_iou_matrix_{}".format(img_idx),
            scope="rpn",
            save_grad=False
        )
        get_tensor_saver().save(
            tensor=torch.transpose(match_quality_matrix, 1, 0),
            tensor_name="CHECK_POINT_iou_matrix_transposed{}".format(img_idx),
            scope="rpn",
            save_grad=False
        )
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for img_idx, (anchors_per_image, targets_per_image) in enumerate(zip(anchors, targets)):
            matched_targets = self.match_targets_to_anchors(
                img_idx, anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")

            get_tensor_saver().save(
                tensor=matched_idxs,
                tensor_name="CHECK_POINT_matched_idxs_img_{}".format(img_idx),
                scope="rpn",
                save_grad=False
            )

            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            get_tensor_saver().save(
                tensor=regression_targets_per_image,
                tensor_name="CHECK_POINT_regression_targets",
                scope="rpn",
                im_idx=img_idx,
                save_grad=False
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        for img_idx, anchors_per_img in enumerate(anchors):
            get_tensor_saver().save(
                tensor=anchors_per_img.bbox,
                tensor_name="concated_anchors_img_{}".format(img_idx),
                scope="rpn",
                save_grad=False
            )
        labels, regression_targets = self.prepare_targets(anchors, targets)

        for img_idx, labels_per_img in enumerate(labels):
            get_tensor_saver().save(
                tensor=labels_per_img,
                tensor_name="labels_img_{}".format(img_idx),
                scope="rpn",
                save_grad=False
            )

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        num_img = len(sampled_pos_inds)
        for img_idx in range(num_img):
            get_tensor_saver().save(
                tensor=torch.nonzero(sampled_pos_inds[img_idx]).squeeze(1),
                tensor_name="CHECK_POINT_sampled_pos_inds_img_{}".format(img_idx),
                scope="rpn",
                save_grad=False
            )
            get_tensor_saver().save(
                tensor=torch.nonzero(sampled_neg_inds[img_idx]).squeeze(1),
                tensor_name="CHECK_POINT_sampled_neg_inds_img_{}".format(img_idx),
                scope="rpn",
                save_grad=False
            )

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        get_tensor_saver().save(
            tensor=sampled_pos_inds,
            tensor_name="sampled_pos_inds",
            scope="rpn",
            save_grad=False
        )

        get_tensor_saver().save(
            tensor=sampled_neg_inds,
            tensor_name="sampled_neg_inds",
            scope="rpn",
            save_grad=False
        )

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        bbox_pred = box_regression[sampled_pos_inds]
        bbox_target = regression_targets[sampled_pos_inds]
        get_tensor_saver().save(
            tensor=bbox_pred,
            tensor_name="CHECK_POINT_bbox_pred",
            scope="rpn",
            save_grad=True
        )

        get_tensor_saver().save(
            tensor=bbox_target,
            tensor_name="CHECK_POINT_bbox_target",
            scope="rpn",
            save_grad=False
        )

        box_loss = smooth_l1_loss(
            bbox_pred,
            bbox_target,
            beta=1.0 / 9,
            size_average=False,
            raw_loss=True,
        )
        get_tensor_saver().save(
            tensor=box_loss,
            tensor_name="CHECK_POINT_rpn_box_reg_loss",
            scope="rpn",
            save_grad=True
        )
        box_loss = box_loss.sum()
        get_tensor_saver().save(
            tensor=box_loss,
            tensor_name="CHECK_POINT_rpn_box_reg_loss_sum",
            scope="rpn",
            save_grad=True
        )
        box_loss /= sampled_inds.numel()
        get_tensor_saver().save(
            tensor=box_loss,
            tensor_name="CHECK_POINT_rpn_box_reg_loss_mean",
            scope="rpn",
            save_grad=True
        )

        get_tensor_saver().save(
            tensor=labels[sampled_inds],
            tensor_name="CHECK_POINT_rpn_cls_labels",
            scope="rpn",
            save_grad=False
        )

        cls_logit = objectness[sampled_inds]
        get_tensor_saver().save(
            tensor=cls_logit,
            tensor_name="CHECK_POINT_rpn_cls_logits",
            scope="rpn",
            save_grad=True
        )

        objectness_loss = F.binary_cross_entropy_with_logits(
            cls_logit, labels[sampled_inds]
        )

        return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.RPN.POSITIVE_FRACTION,
        cfg.ONEFLOW_PYTORCH_COMPARING.RPN_RANDOM_SAMPLE
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
