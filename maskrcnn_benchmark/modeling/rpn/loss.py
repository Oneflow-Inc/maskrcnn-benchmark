# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

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

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
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

    def match_targets_to_anchors(self, anchor, target):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields([])
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
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
                example:
                [
                   [BoxList(num_boxes=163200, image_width=1066, image_height=800, mode=xyxy),
                    BoxList(num_boxes=40800, image_width=1066, image_height=800, mode=xyxy), 
                    BoxList(num_boxes=10200, image_width=1066, image_height=800, mode=xyxy), 
                    BoxList(num_boxes=2550, image_width=1066, image_height=800, mode=xyxy), 
                    BoxList(num_boxes=663, image_width=1066, image_height=800, mode=xyxy)]
                ]
            objectness (list[Tensor])
                example:
                list of <class 'torch.Tensor'>, len(objectness) == 5
                shape of each torch.Tensor:
                [(1, 3, 200, 272),
                 (1, 3, 100, 136),
                 (1, 3, 50, 68),
                 (1, 3, 25, 34),
                 (1, 3, 13, 17)]
            box_regression (list[Tensor])
                example:
                list of <class 'torch.Tensor'>, len(bbox_regression) == 5
                [(1, 12, 200, 272),
                 (1, 12, 100, 136),
                 (1, 12, 50, 68),
                 (1, 12, 25, 34),
                 (1, 12, 13, 17)]
            targets (list[BoxList])
                example:
                [BoxList(num_boxes=23, image_width=1066, image_height=800, mode=xyxy)]
                
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]

        labels, regression_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        for i, (labels_per_im, regression_targets_per_im) in enumerate(zip(labels, regression_targets)):
            get_tensor_saver().save(labels_per_im, 'class_labels', 'rpn', im_idx=i)
            get_tensor_saver().save(regression_targets_per_im, 'regression_targets', 'rpn', im_idx=i)

        for i, (sampled_pos_inds_per_im, sampled_neg_inds_per_im) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            get_tensor_saver().save(sampled_pos_inds_per_im, 'sampled_pos_inds', 'rpn', im_idx=i)
            get_tensor_saver().save(sampled_neg_inds_per_im, 'sampled_neg_inds', 'rpn', im_idx=i)

        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness_flattened = []
        box_regression_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level in zip(
            objectness, box_regression
        ):
            N, A, H, W = objectness_per_level.shape
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )
            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)

        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        get_tensor_saver().save(box_regression, 'box_regression', 'rpn')
        get_tensor_saver().save(regression_targets, 'regression_targets', 'rpn')

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_inds.numel())

        get_tensor_saver().save(box_loss, 'rpn_bbox_reg_loss', 'rpn')

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )

        return objectness_loss, box_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION, cfg.MODEL.RPN.RANDOM_SAMPLE
    )

    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator
