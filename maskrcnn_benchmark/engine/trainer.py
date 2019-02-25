# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger

import numpy as np
import os

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
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        def save_tensor(path, name, tensor_or_tuple):
            if type(tensor_or_tuple) is tuple:
                for idx,i in enumerate(tensor_or_tuple):
                    if i is not None:
                        print(i.size())
                        np.save(path+str(iteration-1)+"."+name+"_"+str(idx)+"."+str(i.size()), i.detach().cpu().numpy())
            else:
                np.save(path+str(iteration-1)+"."+name+"."+str(tensor_or_tuple.size()), tensor_or_tuple.detach().cpu().numpy())
        def fw_callback(module, input, output):
          save_tensor(".", "in", input)
          save_tensor(".", "out", output)
          return
        def bw_callback(module, grad_input, grad_output):
          save_tensor(".", "in_diff", grad_input)
          save_tensor(".", "out_diff", grad_output)
          return
        def register_callback_rec(model, mask_rcnn_root_dir, names, prefix="dump"):
            for (n, m) in model.named_children():
                new_prefix = prefix + "/" + n
                if new_prefix in names:
                    print("registering callback for " + new_prefix)
                    abs_path = os.path.abspath(os.join(mask_rcnn_root_dir, new_prefix))
                    if not os.path.exists(abs_path):
                        os.mkdir(abs_path)
                    os.chdir(abs_path)
                    m.register_forward_hook(fw_callback)
                    m.register_backward_hook(bw_callback)
                    os.chdir(mask_rcnn_root_dir)
                register_callback_rec(m, mask_rcnn_root_dir, names, new_prefix)
        if iteration is 0:
            mask_rcnn_root_dir = os.getcwd()
            register_callback_rec(model, mask_rcnn_root_dir, ["/roi_heads/mask/feature_extractor/pooler/poolers/0"])
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
