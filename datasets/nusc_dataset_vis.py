# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import pickle

import torch
import numpy as np
import PIL.Image as pil
from pyquaternion import Quaternion

from .mono_dataset import MonoDataset


class NuscDatasetVis(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NuscDatasetVis, self).__init__(*args, **kwargs)

        self.data_path = os.path.join(self.opt.dataroot)

        with open(os.path.join(self.opt.dataroot, 'nuscenes_infos_vis_occnerf.pkl'), 'rb') as f:
            self.frame_datas = pickle.load(f)['frames']
        self.filenames = [str(i) for i in range(len(self.frame_datas))]

    def get_info(self, inputs, index_temporal, do_flip):
        inputs[("color", 0, -1)] = []
        inputs[('K_ori', 0)] = []
        inputs["pose_spatial"] = []
        inputs['width_ori'], inputs['height_ori'], inputs['id'] = [], [], []

        idx = int(index_temporal)
        frame_data = self.frame_datas[idx]
        inputs['scene_name'] = [frame_data['scene_name']]
        inputs['frame_idx'] = [frame_data['frame_idx']]

        
        camera_ids = ['front', 'front_left', 'back_left', 'back', 'back_right', 'front_right']
        camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']
        
        cam_num = 6
        for cam_idx in range(cam_num):
            inputs['id'].append(camera_ids[cam_idx])
            cam_sample = frame_data[camera_names[cam_idx]]
            color = self.loader(os.path.join(self.data_path, cam_sample['filename']))
            inputs['width_ori'].append(color.size[0])
            inputs['height_ori'].append(color.size[1])

            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            inputs[("color", 0, -1)].append(color)
            inputs["pose_spatial"].append(cam_sample['cam2ego'].astype(np.float32))

            K = np.eye(4).astype(np.float32)
            K[:3, :3] = cam_sample['intrinsics']
            inputs[('K_ori', 0)].append(K)

        for idx, i in enumerate(self.frame_idxs):
            inputs[('K_ori', 0)] = np.stack(inputs[('K_ori', 0)], axis=0)

        inputs['pose_spatial'] = np.stack(inputs['pose_spatial'], axis=0)

        if 'semantic' in inputs.keys():
            inputs['semantic'] = np.stack(inputs['semantic'], axis=0)

        for key in ['width_ori', 'height_ori']:
            inputs[key] = np.stack(inputs[key], axis=0)

        return


def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat
