# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import numpy
import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        mask_save_dir = './new_dump/mask'
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

            for i, proposals_per_im in enumerate(proposals):
                boxes = proposals_per_im.bbox
                labels = proposals_per_im.get_field('labels')
                boxes_save_path = mask_save_dir + '/{}_proposals_boxes'.format(i) + '.' + str(boxes.size())
                numpy.save(boxes_save_path, boxes.cpu().detach().numpy())
                labels_save_path = mask_save_dir + '/{}_proposals_labels'.format(i) + '.' + str(labels.size())
                numpy.save(labels_save_path, labels.cpu().detach().numpy())

        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)

        mask_logits_save_path = mask_save_dir + '/mask_logits' + '.' + str(mask_logits.size())
        numpy.save(mask_logits_save_path, mask_logits.cpu().detach().numpy())

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        loss_save_path = mask_save_dir + '/loss' + '.' + str(loss_mask.size())
        numpy.save(loss_save_path, loss_mask.cpu().detach().numpy())

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg):
    return ROIMaskHead(cfg)
