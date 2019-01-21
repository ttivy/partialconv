###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Conv2d):
    r"""
    Args:
        multi_channel (bool, optional): Mask each channel. Default: ``False``
        return_mask (bool, optional): Return mask tensor. Default: ``False``
    """

    def __init__(self, *args,
                 multi_channel=False,
                 return_mask=False,
                 **kwargs):

        super().__init__(*args, **kwargs)

        # whether the mask is multi-channel or not
        self.multi_channel = multi_channel
        self.return_mask = return_mask

        if self.multi_channel:
            mask_weight = torch.ones_like(self.weight)
        else:
            mask_weight = torch.ones(1, 1, *self.weight.shape[2:])
        self.register_buffer('mask_weight', mask_weight)

        self.slide_winsize = torch.prod(torch.tensor(self.mask_weight.shape[1:])).item()

        self.last_size = None
        self.mask_output = None
        self.mask_ratio = None

    def forward(self, input, mask=None):
        input_size = input.shape[2:]

        if mask is not None or self.last_size != input_size:
            self.last_size = input_size

            with torch.no_grad():
                if mask is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones_like(input)
                    else:
                        mask = torch.ones(1, 1, *input_size).to(input)

                mask_count = F.conv2d(mask, self.mask_weight, bias=None,
                                      stride=self.stride, padding=self.padding,
                                      dilation=self.dilation, groups=1)
                self.mask_output = torch.gt(mask_count, 0).to(mask_count)
                self.mask_ratio = torch.mul(self.slide_winsize / mask_count.add(1e-8), self.mask_output)

        if self.mask_output.type() != input.type() or self.mask_ratio.type() != input.type():
            self.mask_output = self.mask_output.to(input)
            self.mask_ratio = self.mask_ratio.to(input)

        output = F.conv2d(torch.mul(input, mask) if mask is not None else input,
                          self.weight, bias=None,
                          stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = torch.mul(output, self.mask_ratio)

        if self.bias is not None:
            output = output + self.bias.view(1, self.out_channels, *(1,)*len(input_size))

        output = torch.mul(output, self.mask_output)

        if self.return_mask:
            return output, self.mask_output
        else:
            return output
