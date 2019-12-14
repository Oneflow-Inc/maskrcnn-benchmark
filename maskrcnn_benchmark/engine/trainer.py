# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

from apex import amp

import numpy as np
import pandas as pd
import os
import pickle as pkl

from maskrcnn_benchmark.utils.tensor_saver import create_tensor_saver
from maskrcnn_benchmark.utils.tensor_saver import get_tensor_saver
from maskrcnn_benchmark.utils.tensor_saver import create_mock_data_maker
from maskrcnn_benchmark.utils.tensor_saver import get_mock_data_maker
from maskrcnn_benchmark.utils.tensor_saver import enable_tensor_saver


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
    cfg,
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

    create_tensor_saver(
        training=True,
        base_dir="train_dump",
        iteration=start_iter,
        max_iter=start_iter
        + cfg.ONEFLOW_PYTORCH_COMPARING.MAX_SAVE_TENSOR_ITERATION,
        save_shape=cfg.ONEFLOW_PYTORCH_COMPARING.SAVE_TENSOR_INCLUDE_SHAPE_IN_NAME,
        enable_save=cfg.ONEFLOW_PYTORCH_COMPARING.ENABLE_TENSOR_SAVER,
    )
    create_mock_data_maker(start_iter, enable=False)

    metrics = pd.DataFrame(
        {"iter": 0, "legend": "cfg", "note": str(cfg)}, index=[0]
    )
    for iteration, (images, targets, image_id) in enumerate(
        data_loader, start_iter
    ):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        get_tensor_saver().step()

        scheduler.step()

        if cfg.ONEFLOW_PYTORCH_COMPARING.FAKE_IMAGE_DATA_PATH != "":
            fake_image_path = os.path.join(
                cfg.ONEFLOW_PYTORCH_COMPARING.FAKE_IMAGE_DATA_PATH,
                "image_{}.npy".format(iteration),
            )
            fake_images = np.load(fake_image_path)
            fake_images = np.transpose(fake_images, (0, 3, 1, 2))
            images.tensors = torch.tensor(fake_images)
            logger.info(
                "Load fake image data from {} at itor {}".format(
                    fake_image_path, iteration
                )
            )
        elif cfg.ONEFLOW_PYTORCH_COMPARING.SAVE_IMAGE_TENSOR is True:
            with enable_tensor_saver() as saver:
                saver.save(
                    tensor=images.tensors.permute(0, 2, 3, 1),
                    tensor_name="image",
                )

        get_mock_data_maker().step()
        get_mock_data_maker().update_image(image_id, images)
        get_mock_data_maker().update_target(targets)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        i = iteration - 1
        df = pd.DataFrame(
            [
                {
                    "iter": i,
                    "legend": "elapsed_time",
                    "value": meters.meters["time"].median,
                },
                {
                    "iter": i,
                    "legend": "loss_rpn_box_reg",
                    "value": meters.meters["loss_rpn_box_reg"].median,
                },
                {
                    "iter": i,
                    "legend": "loss_objectness",
                    "value": meters.meters["loss_objectness"].median,
                },
                {
                    "iter": i,
                    "legend": "loss_box_reg",
                    "value": meters.meters["loss_box_reg"].median,
                },
                {
                    "iter": i,
                    "legend": "loss_classifier",
                    "value": meters.meters["loss_classifier"].median,
                },
                {
                    "iter": i,
                    "legend": "loss_mask",
                    "value": meters.meters["loss_mask"].median,
                },
                {
                    "iter": i,
                    "legend": "lr",
                    "value": optimizer.param_groups[0]["lr"],
                },
                {
                    "iter": i,
                    "legend": "max_mem",
                    "value": torch.cuda.max_memory_allocated()
                    / 1024.0
                    / 1024.0,
                },
                {"iter": i, "legend": "loader_time", "value": data_time},
                {
                    "iter": i,
                    "legend": "total_pos_inds_elem_cnt",
                    "value": meters.meters["total_pos_inds_elem_cnt"].median,
                },
            ]
        )
        metrics = pd.concat([metrics, df], axis=0, sort=False)
        if (
            iteration % cfg.ONEFLOW_PYTORCH_COMPARING.METRICS_PERIODS == 0
            or iteration == max_iter
        ):
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
        if (
            iteration % cfg.ONEFLOW_PYTORCH_COMPARING.METRICS_SAVE_CSV_PERIODS
            == 0
            or iteration == max_iter
        ):
            if get_world_size() < 2 or dist.get_rank() == 0:
                npy_file_name = "torch-{}-batch_size-{}-image_dir-{}-{}.csv".format(
                    i,
                    cfg.SOLVER.IMS_PER_BATCH,
                    ":".join(cfg.DATASETS.TRAIN),
                    str(datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")),
                )
                metrics.to_csv(npy_file_name, index=False)
                print("saved: {}".format(npy_file_name))
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

            if cfg.ONEFLOW_PYTORCH_COMPARING.DUMP_MOMENTUM_BUFFER:
                state_dict = optimizer.state_dict()
                model_name2momentum_buffer = {}
                for key, value in model.named_parameters():
                    if value.requires_grad:
                        momentum_buffer = (
                            state_dict["state"][id(value)]["momentum_buffer"]
                            .cpu()
                            .detach()
                            .numpy()
                        )
                        model_name2momentum_buffer[key] = momentum_buffer

                pkl.dump(
                    model_name2momentum_buffer,
                    open(
                        "model_final" + ".model_name2momentum_buffer.pkl", "wb"
                    ),
                    protocol=2,
                )

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
