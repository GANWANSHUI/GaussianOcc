# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import pdb
# from pytorch3d import transforms
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle
from collections import OrderedDict


class PoseDecoder(nn.Module):

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=False)

        self.net = nn.ModuleList(list(self.convs.values()))


        # input_channel= 256 # + 6

        input_channel = 1280 + 6

        pose_width = 256
        pose_depth = 5

        # if refine no relu
        # self.refine_mlp = nn.Sequential(nn.Linear(input_channel, pose_width), nn.ReLU(inplace=True),
        #                                 *[nn.Sequential(nn.Linear(pose_width, pose_width))
        #                             for _ in range(pose_depth)], nn.Linear(pose_width, self.num_frames_to_predict_for*6), nn.ReLU(inplace=True))

        self.refine_mlp = nn.Sequential(nn.Linear(input_channel, pose_width), nn.ReLU(inplace=True),
                                        *[nn.Sequential(nn.Linear(pose_width, pose_width))
                                    for _ in range(pose_depth)], nn.Linear(pose_width, self.num_frames_to_predict_for*6))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.finial_conv = nn.Conv2d(256, 128, 1)

        # if no refine + relu


    def forward(self, input_features, joint_pose=False, coarse_RT=None):

        # pdb.set_trace()
        B, C, H, W = input_features[0][-1].shape
        if joint_pose:
            last_features = [f[-1].reshape(-1, 6, C, H, W).mean(1) for f in input_features]
        else:
            last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features # torch.Size([1, 256, 11, 21])

        # pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu114_pyt1120/download.html

        # 这里拼接 coarse pose and the image feature
        if coarse_RT is not None:
            # coarse_axisangle = transforms.matrix_to_axis_angle(coarse_RT[:3,:3])
            coarse_axisangle = matrix_to_axis_angle(coarse_RT[:3,:3])
            coarse_translation = coarse_RT[:3, 3]
            cat_rt = torch.cat([coarse_axisangle, coarse_translation], 0)

            # pdb.set_trace()

            out = self.pool1(out)
            out = self.finial_conv(out)
            out = self.pool2(out)

            out=out.view(-1, 1280)

            # out = out.mean(3).mean(2)
            out = torch.cat([out, cat_rt.unsqueeze(0)], 1)
            out = self.refine_mlp(out)
            out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

            # gt
            # out[..., :3][:, 0] = coarse_axisangle
            # out[..., 3:][:, 0] = coarse_translation

            # pdb.set_trace()
            # residual refine 
            # out[..., :3] += coarse_axisangle
            # out[..., 3:] += coarse_translation

            # axisangle = out[..., :3] 
            # translation = out[..., 3:] 

            axisangle = out[..., :3] # + coarse_axisangle
            translation = out[..., 3:] + coarse_translation


        else:

            for i in range(3):

                out = self.convs[("pose", i)](out)

                if i != 2:

                    out = self.relu(out)

            out = out.mean(3).mean(2)

            out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)


            axisangle = out[..., :3]
            translation = out[..., 3:]

        # residual 

        return axisangle, translation
