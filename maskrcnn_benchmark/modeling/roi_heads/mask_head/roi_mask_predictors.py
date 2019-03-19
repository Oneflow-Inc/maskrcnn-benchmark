# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


class MaskRCNNC4Predictor(nn.Module):
    def __init__(self, cfg):
        super(MaskRCNNC4Predictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        dim_reduced = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[-1]

        if cfg.MODEL.ROI_HEADS.USE_FPN:
            num_inputs = dim_reduced
        else:
            stage_index = 4
            stage2_relative_factor = 2 ** (stage_index - 1)
            res2_out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
            num_inputs = res2_out_channels * stage2_relative_factor

        self.conv5_mask = ConvTranspose2d(num_inputs, dim_reduced, 2, 2, 0)
        self.mask_fcn_logits = Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        mask_conv5_in_dump_path = './new_dump/mask/conv5_in' + '.' + str(x.size())
        numpy.save(mask_conv5_in_dump_path, x.cpu().detach().numpy())
        mask_conv5_in_grad_dump_path = './new_dump/mask/conv5_in_grad' + '.' + str(x.size())
        x.register_hook(lambda grad : numpy.save(mask_conv5_in_grad_dump_path, grad.cpu().detach().numpy()))

        x = F.relu(self.conv5_mask(x))

        mask_conv5_out_dump_path = './new_dump/mask/conv5_out' + '.' + str(x.size())
        numpy.save(mask_conv5_out_dump_path, x.cpu().detach().numpy())
        mask_conv5_out_grad_dump_path = './new_dump/mask/conv5_out_grad' + '.' + str(x.size())
        x.register_hook(lambda grad : numpy.save(mask_conv5_out_grad_dump_path, grad.cpu().detach().numpy()))

        return self.mask_fcn_logits(x)


_ROI_MASK_PREDICTOR = {"MaskRCNNC4Predictor": MaskRCNNC4Predictor}


def make_roi_mask_predictor(cfg):
    func = _ROI_MASK_PREDICTOR[cfg.MODEL.ROI_MASK_HEAD.PREDICTOR]
    return func(cfg)
