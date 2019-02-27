# -*- coding: utf-8 -*

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

import maskrcnn_benchmark

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
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        # 由于不同module的输入输出类型不同，save_tensor保存blob的策略一边验证一边修改
        # 等全网络验证完成之后会产生一个比较好的存储blob的目录结构
        def save_tensor(path, name, entity_to_save):
            if iteration - 1 == 0:
                if type(entity_to_save) is tuple:
                    print 'tuple!!!'
                    for idx, item in enumerate(entity_to_save):
                        # tuple of lists
                        if (type(item) is list):
                            print '\t tuple of lists'
                            for idx_2, elem in enumerate(item):
                                # tuple of lists of Tensors
                                if (type(item) is torch.Tensor):
                                    np.save(path + "/" + "iter-" + str(iteration-1) + name  + "_" + str(idx) + "_" + str(idx_2) + "." + str(item.size()), item.detach().cpu().numpy())
                        # tuple of ImageList 
                        elif (type(item) is maskrcnn_benchmark.structures.image_list.ImageList):
                            print '\t tuple of ImageList'
                        # tuple of Tensors
                        elif (type(item) is torch.Tensor):
                            print '\t tuple of Tensors'
                            np.save(path + "/" + "iter-" + str(iteration - 1) + "." + name + "_" + str(idx) + "." + str(item.size()), item.detach().cpu().numpy())
                        elif (type(item) is dict):
                            print '\t tuple of dicts'
                        else:
                            print '\t type' + str(type(item)) + 'does not support!'
                            assert False
                elif type(entity_to_save) is list:
                    print 'list!!!'
                    for idx, item in enumerate(entity_to_save):
                        # list of Tensors
                        if (type(iter) is torch.Tensor):
                            print '\t list of Tensors'
                elif type(entity_to_save) is torch.Tensor:
                    print 'torch.Tensor!!!'
                    np.save(path + "/" + "iter-" + str(iteration - 1) + "." + name + "." + str(entity_to_save.size()), entity_to_save.detach().cpu().numpy())
                else:
                    assert False
        def fw_callback(module, input, output):
            module_name = module2name[module]
            print 'We are in ' + module_name + "'s fw_callback function."
            path = 'dump' + module_name
            if not os.path.exists(path):
                os.makedirs(path)
            if iteration - 1 == 0:
                save_tensor(path, "in", input)
                save_tensor(path, "out", output)
            return
        def bw_callback(module, grad_input, grad_output):
            module_name = module2name[module]
            print 'We are in ' + module_name + "'s fw_callback function."
            path = 'dump' + module_name
            if not os.path.exists(path):
                os.makedirs(path)
            if iteration - 1 == 0:
                save_tensor(path, "in_diff", grad_input)
                save_tensor(path, "out_diff", grad_output)
            return
        def register_callback_rec_for_all_modules(module, prefix=""):
            for (n, m) in module.named_children():
                new_prefix = prefix + "/" + n
                module2name[m] = new_prefix
                print new_prefix
                # print("registering callback for " + new_prefix)
                m.register_forward_hook(fw_callback)
                m.register_backward_hook(bw_callback)
                register_callback_rec_for_all_modules(m, new_prefix)
        def register_callback_rec_for_particular_modules(module, names, prefix=""):
            for (n, m) in module.named_children():
                new_prefix = prefix + "/" + n
                if n in names or new_prefix in names:
                    module2name[m] = new_prefix
                    print("registering callback for " + new_prefix)
                    m.register_forward_hook(fw_callback)
                    m.register_backward_hook(bw_callback)
                register_callback_rec_for_particular_modules(m, names, new_prefix)
        # save modules' in, out, in_diff, out_diff
        # if iteration is 0:
        #     register_callback_rec_for_particular_modules(model, ['/backbone', '/rpn', '/roi_heads'])

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
