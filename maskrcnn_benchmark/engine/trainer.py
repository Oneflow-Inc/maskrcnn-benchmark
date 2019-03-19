# -*- coding: utf-8 -*

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

import maskrcnn_benchmark
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.structures.bounding_box import BoxList

import numpy as np
import os

from functools import partial

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    module2name = {}

    save_dir = './new_dump'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    register_param_grad_hook(model)

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        if iteration == start_iter:
            if arguments["fake_image"]:
                fake_images = np.load(arguments["fake_image"])
                fake_images = np.transpose(fake_images, (0, 3, 1, 2))
                images.tensors = torch.tensor(fake_images)
                logger.info("Load fake image data from {} at itor {}".format(arguments["fake_image"], iteration))
            else:
                first_images_save_path = save_dir + '/images' + '.' + str(images.tensors.size())
                np.save(first_images_save_path, images.tensors.cpu().detach().numpy())

        data_time = time.time() - end

        iteration = iteration + 1
        arguments["iteration"] = iteration

        scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

def register_param_grad_hook(model):
    param_grad_dump_dir = './param_grad'
    if not os.path.exists(param_grad_dump_dir):
        os.makedirs(param_grad_dump_dir)

    def dump_param_grad(dump_path, param_grad):
        param_grad_dump_path = dump_path + '.' + str(param_grad.size())
        np.save(param_grad_dump_path, param_grad.detach().cpu().numpy())

    def get_dump_path(param_name):
        param_grad_name = param_name.replace('.weight', '.weight_grad')
        param_grad_name = param_name.replace('.bias', '.bias_grad')
        param_grad_name = param_grad_name.replace('.', '-')
        return os.path.join(param_grad_dump_dir, param_grad_name)

    for key, value in model.named_parameters():
        if value.requires_grad:
            value.register_hook(partial(dump_param_grad, get_dump_path(key)))

# xfjiang: save param grad
# for key, value in model.named_parameters():
#     print(key)
#     backbone.body.stem.conv1.weight
#     backbone.body.layer1.0.downsample.0.weight
#     backbone.body.layer1.0.conv1.weight
#     backbone.body.layer1.0.conv2.weight
#     backbone.body.layer1.0.conv3.weight
#     backbone.body.layer1.1.conv1.weight
#     backbone.body.layer1.1.conv2.weight
#     backbone.body.layer1.1.conv3.weight
#     backbone.body.layer1.2.conv1.weight
#     backbone.body.layer1.2.conv2.weight
#     backbone.body.layer1.2.conv3.weight
#     backbone.body.layer2.0.downsample.0.weight
#     backbone.body.layer2.0.conv1.weight
#     backbone.body.layer2.0.conv2.weight
#     backbone.body.layer2.0.conv3.weight
#     backbone.body.layer2.1.conv1.weight
#     backbone.body.layer2.1.conv2.weight
#     backbone.body.layer2.1.conv3.weight
#     backbone.body.layer2.2.conv1.weight
#     backbone.body.layer2.2.conv2.weight
#     backbone.body.layer2.2.conv3.weight
#     backbone.body.layer2.3.conv1.weight
#     backbone.body.layer2.3.conv2.weight
#     backbone.body.layer2.3.conv3.weight
#     backbone.body.layer3.0.downsample.0.weight
#     backbone.body.layer3.0.conv1.weight
#     backbone.body.layer3.0.conv2.weight
#     backbone.body.layer3.0.conv3.weight
#     backbone.body.layer3.1.conv1.weight
#     backbone.body.layer3.1.conv2.weight
#     backbone.body.layer3.1.conv3.weight
#     backbone.body.layer3.2.conv1.weight
#     backbone.body.layer3.2.conv2.weight
#     backbone.body.layer3.2.conv3.weight
#     backbone.body.layer3.3.conv1.weight
#     backbone.body.layer3.3.conv2.weight
#     backbone.body.layer3.3.conv3.weight
#     backbone.body.layer3.4.conv1.weight
#     backbone.body.layer3.4.conv2.weight
#     backbone.body.layer3.4.conv3.weight
#     backbone.body.layer3.5.conv1.weight
#     backbone.body.layer3.5.conv2.weight
#     backbone.body.layer3.5.conv3.weight
#     backbone.body.layer4.0.downsample.0.weight
#     backbone.body.layer4.0.conv1.weight
#     backbone.body.layer4.0.conv2.weight
#     backbone.body.layer4.0.conv3.weight
#     backbone.body.layer4.1.conv1.weight
#     backbone.body.layer4.1.conv2.weight
#     backbone.body.layer4.1.conv3.weight
#     backbone.body.layer4.2.conv1.weight
#     backbone.body.layer4.2.conv2.weight
#     backbone.body.layer4.2.conv3.weight
#     backbone.fpn.fpn_inner1.weight
#     backbone.fpn.fpn_inner1.bias
#     backbone.fpn.fpn_layer1.weight
#     backbone.fpn.fpn_layer1.bias
#     backbone.fpn.fpn_inner2.weight
#     backbone.fpn.fpn_inner2.bias
#     backbone.fpn.fpn_layer2.weight
#     backbone.fpn.fpn_layer2.bias
#     backbone.fpn.fpn_inner3.weight
#     backbone.fpn.fpn_inner3.bias
#     backbone.fpn.fpn_layer3.weight
#     backbone.fpn.fpn_layer3.bias
#     backbone.fpn.fpn_inner4.weight
#     backbone.fpn.fpn_inner4.bias
#     backbone.fpn.fpn_layer4.weight
#     backbone.fpn.fpn_layer4.bias
#     rpn.head.conv.weight
#     rpn.head.conv.bias
#     rpn.head.cls_logits.weight
#     rpn.head.cls_logits.bias
#     rpn.head.bbox_pred.weight
#     rpn.head.bbox_pred.bias
#     roi_heads.box.feature_extractor.fc6.weight
#     roi_heads.box.feature_extractor.fc6.bias
#     roi_heads.box.feature_extractor.fc7.weight
#     roi_heads.box.feature_extractor.fc7.bias
#     roi_heads.box.predictor.cls_score.weight
#     roi_heads.box.predictor.cls_score.bias
#     roi_heads.box.predictor.bbox_pred.weight
#     roi_heads.box.predictor.bbox_pred.bias
#     roi_heads.mask.feature_extractor.mask_fcn1.weight
#     roi_heads.mask.feature_extractor.mask_fcn1.bias
#     roi_heads.mask.feature_extractor.mask_fcn2.weight
#     roi_heads.mask.feature_extractor.mask_fcn2.bias
#     roi_heads.mask.feature_extractor.mask_fcn3.weight
#     roi_heads.mask.feature_extractor.mask_fcn3.bias
#     roi_heads.mask.feature_extractor.mask_fcn4.weight
#     roi_heads.mask.feature_extractor.mask_fcn4.bias
#     roi_heads.mask.predictor.conv5_mask.weight
#     roi_heads.mask.predictor.conv5_mask.bias
#     roi_heads.mask.predictor.mask_fcn_logits.weight
#     roi_heads.mask.predictor.mask_fcn_logits.bias
