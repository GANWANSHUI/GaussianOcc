# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import random
import numpy as np

import cv2
import torch
import open3d as o3d
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders
    """
    def __init__(self,
                 opt,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 volume_depth= False,
                 **kwargs):
        super(MonoDataset, self).__init__()

        self.opt = opt
        self.self_supervise = self.opt.self_supervise
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = transforms.InterpolationMode.LANCZOS

        self.frame_idxs = frame_idxs

        self.frame_idxs_permant = frame_idxs

        self.is_train = is_train
        self.volume_depth = volume_depth

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n + "_aug", im, -1)] = []

                for i in range(self.num_scales):
                    inputs[(n, im, i)] = []
                    inputs[(n + "_aug", im, i)] = []
                    #print(n, im, i)
                    num_cam = len(inputs[(n, im, i - 1)])
                    for index_spatial in range(num_cam):
                        inputs[(n, im, i)].append(self.resize[i](inputs[(n, im, i - 1)][index_spatial]))


        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                num_cam = len(f)
                for index_spatial in range(num_cam):
                    aug = color_aug(f[index_spatial])
                    # try:
                    inputs[(n, im, i)][index_spatial] = self.to_tensor(f[index_spatial])
                    inputs[(n + "_aug", im, i)].append(self.to_tensor(aug))
                
                inputs[(n, im, i)] = torch.stack(inputs[(n, im, i)], dim=0)
                inputs[(n + "_aug", im, i)] = torch.stack(inputs[(n + "_aug", im, i)], dim=0)

    def __len__(self):
        if self.opt.debug:
            return 10
        else:
            return len(self.filenames)


    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """

        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = False

        frame_index = self.filenames[index].strip().split()[0]

        self.frame_idxs = self.frame_idxs_permant

        self.get_info(inputs, frame_index, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        if not self.is_train and self.opt.use_t == 'No':
            self.frame_idxs = [0]


        for scale in range(self.num_scales):
            for frame_id in self.frame_idxs:
                inputs[("K", frame_id, scale)] = []
                inputs[("inv_K", frame_id, scale)] = []
                inputs[("K_render", frame_id, scale)] = []

        cam_num = 6 * 3 if (self.opt.auxiliary_frame and self.is_train or (self.opt.use_t != 'No')) else 6

        for index_spatial in range(cam_num):
            for scale in range(self.num_scales):
                for frame_id in self.frame_idxs:
                    K = inputs[('K_ori', frame_id)][index_spatial].copy()
                    K[0, :] *= (self.width // (2 ** scale)) / inputs['width_ori'][index_spatial]
                    K[1, :] *= (self.height // (2 ** scale)) / inputs['height_ori'][index_spatial]
                    inv_K = np.linalg.pinv(K)
                    inputs[("K", frame_id, scale)].append(torch.from_numpy(K))
                    inputs[("inv_K", frame_id, scale)].append(torch.from_numpy(inv_K))

                    K_render = inputs[('K_ori', frame_id)][index_spatial].copy()
                    K_render[0, :] *= (self.opt.render_w // (2 ** scale)) / inputs['width_ori'][index_spatial]
                    K_render[1, :] *= (self.opt.render_h // (2 ** scale)) / inputs['height_ori'][index_spatial]
                    inputs[("K_render", frame_id, scale)].append(torch.from_numpy(K_render))


        for scale in range(self.num_scales):
            for frame_id in self.frame_idxs:
                inputs[("K", frame_id, scale)] = torch.stack(inputs[("K", frame_id, scale)], dim=0)
                inputs[("inv_K", frame_id, scale)] = torch.stack(inputs[("inv_K", frame_id, scale)], dim=0)
                inputs[("K_render", frame_id, scale)] = torch.stack(inputs[("K_render", frame_id, scale)], dim=0)


        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = lambda x: x

        self.preprocess(inputs, color_aug)

        del inputs[("color", 0, -1)]
        del inputs['width_ori']
        del inputs['height_ori']

        if 'depth' in inputs.keys():
            inputs['depth'] = torch.from_numpy(inputs['depth'])

        if 'pose_spatial' in inputs.keys():
            inputs["pose_spatial"] = torch.from_numpy(inputs["pose_spatial"])

        if 'semantic' in inputs.keys():
            inputs['semantic'] = torch.from_numpy(inputs['semantic'])

        if 'semantics_3d' in inputs.keys():
            inputs['semantics_3d'] = torch.from_numpy(inputs['semantics_3d']).long()
            inputs['mask_camera_3d'] = torch.from_numpy(inputs['mask_camera_3d']).long()
        
        if 'ego0_T_global' in inputs.keys():
            inputs['ego0_T_global'] = torch.from_numpy(inputs['ego0_T_global'])

        if self.is_train:

            for i in [-1, 1]:
                if ("pose_spatial", i) in inputs.keys():
                    inputs[("pose_spatial", i)] = torch.from_numpy(inputs[("pose_spatial", i)])

            if self.opt.use_fix_mask:

                inputs["mask"] = []

                for i in range(6):
                    temp = cv2.resize(inputs["mask_ori"][i], (self.width, self.height))
                    temp = temp[..., 0]
                    temp = (temp == 0).astype(np.float32)
                    inputs["mask"].append(temp)

                if self.opt.dataset == 'nusc':
                    inputs["mask"] = abs(1 - np.stack(inputs["mask"], axis=0))

                else:

                    inputs["mask"] = np.stack(inputs["mask"], axis=0)
                inputs["mask"] = np.tile(inputs["mask"][:, None], (1, 2, 1, 1))
                inputs["mask"] = torch.from_numpy(inputs["mask"])

                if do_flip:
                    inputs["mask"] = torch.flip(inputs["mask"], [3])

                del inputs["mask_ori"]


        if self.volume_depth and 'gs' not in self.opt.render_type:

            with torch.no_grad():
                rays_o, rays_d = get_rays_of_a_view(H=self.opt.render_h, W=self.opt.render_w, K=inputs[('K_render', 0, 0)], c2w=inputs['pose_spatial'],
                                                           ndc=False, inverse_y=True, flip_x=False, flip_y=False, mode='center', cam_num=cam_num)
                inputs['rays_o', 0] = rays_o
                inputs['rays_d', 0] = rays_d


        inputs["all_cam_center"] = torch.from_numpy(np.array([1.2475059, 0.0673422, 1.5356342])).unsqueeze(0).to(torch.float32) # DDAD

        if self.is_train:
            # if not self.self_supervise:
            #     del inputs['point_cloud']
            #     for i in self.frame_idxs[1:]:
            #         # del inputs[("color", i, -1)]  # TODO
            #         del inputs[("color_aug", i, -1)]
            for i in self.frame_idxs:
                del inputs[('K_ori', i)]

        else:
            del inputs[('K_ori', 0)]
            

        if self.opt.use_t != 'No':
            for i in [-1, 1]:
                if ("pose_spatial", i) in inputs.keys():
                    del inputs[("pose_spatial", i)]

                # print(inputs.keys())
                if ("K_ori", i) in inputs.keys():
                    del inputs[('K_ori', i)]

        if self.opt.dataset == 'nusc':
            for frame_id in self.frame_idxs[1:]:
                inputs[('cam_T_cam', frame_id)] = torch.from_numpy(inputs[('cam_T_cam', frame_id)]).to(torch.float32)

        else:
            if self.is_train:
                del inputs['point_cloud']
                # if 'mask_ori' in inputs.keys():
                #     del inputs['mask_ori']
                del inputs['point_cloud_path']
        
        if 'mask_ori' in inputs.keys():
            
            del inputs['mask_ori']

        return inputs

    def get_info(self, inputs, index, do_flip):
        raise NotImplementedError

    def get_mask(self, pts_xyz):

        mask1 = pts_xyz[..., 2] < 0.001

        mask2 = pts_xyz[..., 2] >= self.opt.real_size[5]

        xy_range = self.opt.max_depth

        # x
        mask3 = pts_xyz[..., 0] > xy_range
        mask4 = pts_xyz[..., 0] < -xy_range

        # y
        mask5 = pts_xyz[..., 1] > xy_range
        mask6 = pts_xyz[..., 1] < -xy_range

        # mask out the point cloud close to car, especially for nuscenes
        # x
        mask7 = (pts_xyz[..., 0] < 3.5) & (pts_xyz[..., 0] > -0.5)
        # y
        mask8 = (pts_xyz[..., 1] < 1.0) & (pts_xyz[..., 1] > -1.0)

        mask9 = mask7 & mask8

        mask = mask1 + mask2 + mask3 + mask4 + mask5 + mask6 + mask9

        return mask

def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center', cam_num=6):

    with torch.no_grad():
        rays_o_all = torch.zeros(cam_num, H, W, 3)
        rays_d_all = torch.zeros(cam_num, H, W, 3)

        for i in range (cam_num):
            rays_o, rays_d = get_rays(H, W, K[i, ...], c2w[i, ...], inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
            rays_o_all[i,...] = rays_o
            rays_d_all[i,...] = rays_d

    return rays_o_all, rays_d_all


def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass

    elif mode == 'center':
        i, j = i + 0.5, j + 0.5

    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1) #

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d
