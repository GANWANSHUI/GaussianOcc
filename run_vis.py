# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import os.path as osp
import json
import time

import cv2
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pickle
import datasets
import networks
from options import MonodepthOptions
from utils.loss_metric import *
from utils.layers import *


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


class DistributedSampler(_DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices = indices[self.rank:self.total_size:self.num_replicas]

        return iter(indices)


class Runer:

    def __init__(self, options):

        self.opt = options
        self.opt.B = self.opt.batch_size // 6
        if self.opt.debug:
            self.opt.voxels_size = [8, 128, 128]
            self.opt.render_h = 45
            self.opt.render_w = 80
            self.opt.num_workers = 1
            self.opt.model_name = "debug/"
        

        from pathlib import Path
        path = Path(self.opt.load_weights_folder)
        new_path = path.parents[1]

        self.log_path =  new_path

        print('log path:', self.log_path)
        os.makedirs(osp.join(self.log_path, 'eval'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'visual_rgb_depth'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'visual_feature'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'scene_video'), exist_ok=True)

        self.models = {}
        
        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend='nccl')
        self.device = torch.device("cuda", self.local_rank)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        # self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models["encoder"] = networks.Encoder_res101(self.opt.input_channel, path=None, network_type=self.opt.encoder)
        self.models["depth"] = networks.VolumeDecoder(self.opt)
        self.log_print('N_samples: {}'.format(self.models["depth"].N_samples))
        self.log_print('Voxel size: {}'.format(self.models["depth"].voxel_size))

        self.models["encoder"] = self.models["encoder"].to(self.device)
        self.models["depth"] = self.models["depth"].to(self.device)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        for key in self.models.keys():
            self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank,
                                   find_unused_parameters=True, broadcast_buffers=False)

        if self.local_rank == 0:
            self.log_print("Training model named: {}".format(self.opt.model_name))

        datasets_dict = {
            "ddad": datasets.DDADDataset,
            "nusc": datasets.NuscDatasetVis}

        self.dataset = datasets_dict[self.opt.dataset]

        self.opt.batch_size = self.opt.batch_size // 6

        val_dataset = self.dataset(self.opt,
                                   self.opt.height, self.opt.width,
                                   [0], num_scales=1, is_train=False,  # the first is frame_ids
                                   volume_depth=self.opt.volume_depth)
        rank, world_size = get_dist_info()
        self.world_size = world_size
        val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, collate_fn=self.my_collate,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False, sampler=val_sampler)

        self.num_val = len(val_dataset)

        self.opt.batch_size = self.opt.batch_size * 6
        self.num_val = self.num_val * 6

        self.save_opts()


    def my_collate(self, batch):
        batch_new = {}
        keys_list = list(batch[0].keys())
        special_key_list = ['id', 'scene_name', 'frame_idx']

        for key in keys_list:
            if key not in special_key_list:
                # print('key:', key)
                batch_new[key] = [item[key] for item in batch]
                try:
                    batch_new[key] = torch.cat(batch_new[key], axis=0)
                except:
                    print('key', key)

            else:
                batch_new[key] = []
                for item in batch:
                    for value in item[key]:
                        # print(value.shape)
                        batch_new[key].append(value)

        return batch_new

    def to_device(self, inputs):

        special_key_list = ['id', ('K_ori', -1), ('K_ori', 1), 'scene_name', 'frame_idx']

        for key, ipt in inputs.items():

            if key in special_key_list:
                inputs[key] = ipt

            else:
                inputs[key] = ipt.to(self.device)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self, save_image=True):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        print('begin eval!')
        total_time = []
        total_evl_time = time.time()

        with torch.no_grad():
            loader = self.val_loader
            for idx, data in enumerate(loader):

                scene_name = data['scene_name'][0]
                frame_idx = data['frame_idx'][0]
                
                if 'ddad' in self.opt.config:
                    # ddad
                    scene_list = ['000159', '000180', '000188', '000194', '000195']
                else:
                    # nuscenes
                    scene_list = ['scene-0012', 'scene-0018', 'scene-0038','scene-0092']

                # if scene_name in scene_list and frame_idx < 100:
                if scene_name in scene_list and frame_idx < 100:

                    eps_time = time.time()

                    input_color = data[("color", 0, 0)].cuda()

                    camera_ids = data["id"]

                    features = self.models["encoder"](input_color)

                    output = self.models["depth"](features, data, is_train=False, no_depth=self.opt.use_semantic)

                    eps_time = time.time() - eps_time
                    total_time.append(eps_time)

                    if self.local_rank == 0 and idx % 100 == 0:
                        print('single inference:(eps time:', eps_time, 'secs)')

                    # if not self.opt.use_semantic:
                    pred_depths = output[("disp", 0)].cpu()[:, 0].numpy()

                    concated_image_list = []
                    concated_depth_list = []

                    for i in range(input_color.shape[0]):

                        camera_id = camera_ids[i]

                        color = (input_color[i].cpu().permute(1, 2, 0).numpy()) * 255
                        color = color[..., [2, 1, 0]]
                        concated_image_list.append(cv2.resize(color.copy(), (320, 180)))
                        
                        # if not self.opt.use_semantic:
                        pred_depth = pred_depths[i]
                        pred_depth_color = visualize_depth(pred_depth.copy())
                        concated_depth_list.append(cv2.resize(pred_depth_color.copy(), (320, 180)))


                    image_left_front_right = np.concatenate(
                        (concated_image_list[1], concated_image_list[0], concated_image_list[5]), axis=1)
                    image_left_rear_right = np.concatenate(
                        (concated_image_list[2], concated_image_list[3], concated_image_list[4]), axis=1)
                    # image_surround_view = np.concatenate((image_left_front_right, image_left_rear_right), axis=0)


                    surround_depth_view_up = np.concatenate(
                            (concated_depth_list[1], concated_depth_list[0], concated_depth_list[5]), axis=1)
                    surround_depth_view_down = np.concatenate(
                            (concated_depth_list[2], concated_depth_list[3], concated_depth_list[4]), axis=1)

                    if not self.opt.use_semantic:
                        surround_view_up = np.concatenate((image_left_front_right, surround_depth_view_up), axis=0)
                        surround_view_down = np.concatenate((image_left_rear_right, surround_depth_view_down), axis=0)
                    
                    else:
                        surround_view_up = image_left_front_right
                        surround_view_down = image_left_rear_right
                    
                    # pdb.set_trace()
                    scene_name = data['scene_name'][0]
                    frame_idx = data['frame_idx'][0]

                    os.makedirs('{}/scene_video/{}'.format(self.log_path, scene_name), exist_ok=True)
                    cv2.imwrite('{}/scene_video/{}/{:03d}-up.jpg'.format(self.log_path, scene_name, frame_idx), surround_view_up)
                    cv2.imwrite('{}/scene_video/{}/{:03d}-down.jpg'.format(self.log_path, scene_name, frame_idx), surround_view_down)

                    cv2.imwrite('{}/scene_video/{}/{:03d}-depth-up.jpg'.format(self.log_path, scene_name, frame_idx), surround_depth_view_up)
                    cv2.imwrite('{}/scene_video/{}/{:03d}-depth-down.jpg'.format(self.log_path, scene_name, frame_idx), surround_depth_view_down)

                    vis_dic = {}
                    # vis_dic['opt'] = self.opt
                    # vis_dic['depth_color'] = concated_depth_list
                    # vis_dic['rgb'] = concated_image_list
                    vis_dic['pose_spatial'] = data['pose_spatial'].detach().cpu().numpy()
                    vis_dic['probability'] = output['pred_occ_logits'].detach().cpu().numpy()
                    np.save('{}/scene_video/{}/{:03d}-out.npy'.format(self.log_path, scene_name, frame_idx), vis_dic)


        eps_time = time.time() - total_evl_time

        print('finish eval!')

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = osp.join(self.log_path, "models")
        if not osp.exists(models_dir):
            os.makedirs(models_dir)
        os.makedirs(osp.join(self.log_path, "eval"), exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(osp.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = osp.expanduser(self.opt.load_weights_folder)

        if self.local_rank == 0:
            assert osp.isdir(self.opt.load_weights_folder), \
                "Cannot find folder {}".format(self.opt.load_weights_folder)
            self.log_print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:

            if self.local_rank == 0:
                self.log_print("Loading {} weights...".format(n))
            path = osp.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

    def log_print(self, str):
        print(str)
        with open(osp.join(self.log_path, 'log.txt'), 'a') as f:
            f.writelines(str + '\n')

    def log_print_train(self, str):
        print(str)
        with open(osp.join(self.log_path, 'log_train.txt'), 'a') as f:
            f.writelines(str + '\n')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    setup_seed(42)
    runner = Runer(opts)
    runner.val()
