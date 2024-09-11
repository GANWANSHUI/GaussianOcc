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
import shutil
import pickle
from copy import deepcopy

import cv2, pdb
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from nuscenes.nuscenes import NuScenes

import datasets
import networks
from utils import occ_metrics
from utils.loss_metric import *
from utils.layers import *

import utils.basic as basic
import datetime, pytz


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
        self.opt.B = self.opt.batch_size // self.opt.cam_N
        
        # pdb.set_trace()
        if 'gs' in self.opt.render_type:
            self.opt.semantic_sample_ratio = 1.0

        if self.opt.debug:
            self.opt.voxels_size = [12, 128, 128]
            self.opt.render_h = 90
            self.opt.render_w = 160
            self.opt.num_workers = 1
            self.opt.model_name = "debug/"

        if 'ddad' in self.opt.config:
            self.opt.max_depth_test = 200
            self.opt.rayiou = 'No'
      
        # pdb.set_trace()
        
        self.log_path = osp.join(self.opt.log_dir, self.opt.model_name + '_' + self.opt.data_type , 'seg_{}_{}_{}_{}_pose_{}_{}_mask_{}'.format(
        self.opt.use_semantic, self.opt.semantic_loss_weight, self.opt.weight_entropy_last, self.opt.weight_distortion,
        self.opt.gt_pose, self.opt.detach_pose, self.opt.use_fix_mask), 
        'type_{}_{}_s_{}_{}_l_{}_en_w_{}_ep_{}_f_{}_infi_{}_cont_{}_depth_{}_{}_{}'.format(
        self.opt.render_type, self.opt.gs_sample, self.opt.gs_scale, self.opt.render_h,  self.opt.self_supervise, self.opt.weight_entropy_last, self.opt.num_epochs,
        self.opt.auxiliary_frame , self.opt.infinite_range, self.opt.contracted_coord, self.opt.min_depth, self.opt.max_depth, self.opt.encoder),
        'exp_{}'.format(datetime.datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime("%Y_%m_%d_%H_%M_%S")))
        
        print('---------------------------------------------------------')
        print('-------------log path:', self.log_path)
        
        os.makedirs(osp.join(self.log_path, 'eval'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'visual_rgb_depth'), exist_ok=True)
        os.makedirs(osp.join(self.log_path, 'visual_feature'), exist_ok=True)


        self.models = {}
        self.parameters_to_train = []

        self.local_rank = self.opt.local_rank
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend='nccl')
        self.device = torch.device("cuda", self.local_rank)

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 # if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        if self.opt.encoder == 'tiny07':

            self.models["encoder"] = networks.NewCRFDepth.NewCRFDepth(version='tiny07', inv_depth=False, max_depth=200, opt = self.opt)

        elif self.opt.encoder == 'small07':

            self.models["encoder"] = networks.NewCRFDepth.NewCRFDepth(version='small07', inv_depth=False, max_depth=200, opt = self.opt)
        
        elif self.opt.encoder == 'large07':
    
            self.models["encoder"] = networks.NewCRFDepth.NewCRFDepth(version='small07', inv_depth=False, max_depth=200, opt = self.opt)
    
        elif self.opt.encoder == '50' or self.opt.encoder == '101':
            self.models["encoder"] = networks.Encoder_res101(self.opt.input_channel, path=None, network_type=self.opt.encoder)
        
        else: 
            print('please define the encoder!')



        self.models["depth"] = networks.VolumeDecoder(self.opt)
        
        
        self.log_print('N_samples: {}'.format(self.models["depth"].N_samples))
        self.log_print('Voxel size: {}'.format(self.models["depth"].voxel_size))


        self.models["encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["encoder"])
        self.models["encoder"] = (self.models["encoder"]).to(self.device)
        self.parameters_to_train += [{'params': self.models["encoder"].parameters(), 'lr': self.opt.learning_rate, 'weight_decay': self.opt.weight_decay}]

        self.models["depth"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["depth"])
        self.models["depth"] = (self.models["depth"]).to(self.device)
        self.parameters_to_train += [{'params': self.models["depth"].parameters(), 'lr': self.opt.de_lr, 'weight_decay': self.opt.weight_decay}]


        # 6d pose
        if self.opt.gt_pose != 'No':

            if self.opt.eval_only:
                self.opt.models_to_load = ['encoder', 'depth', 'pose_encoder', 'pose']
            else:
                self.opt.models_to_load = ['pose_encoder', 'pose']


            self.models["pose_encoder"] = networks.ResnetEncoder(
            34, True, num_input_images=self.num_pose_frames)

            self.models["pose_encoder"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose_encoder"])
            self.models["pose_encoder"] = self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += [{'params': self.models["pose_encoder"].parameters(), 'lr': self.opt.learning_rate}]

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1, num_frames_to_predict_for=2)

            self.models["pose"] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.models["pose"])
            self.models["pose"] = (self.models["pose"]).to(self.device)
            self.parameters_to_train += [{'params': self.models["pose"].parameters(), 'lr': self.opt.learning_rate}]


        if self.opt.load_weights_folder is not None:

            self.load_model()

        # pdb.set_trace()
        
        for key in self.models.keys():
            self.models[key] = DDP(self.models[key], device_ids=[self.local_rank], output_device=self.local_rank,
                                   find_unused_parameters=True, broadcast_buffers=False)

        self.model_optimizer = optim.AdamW(self.parameters_to_train)
        self.criterion = nn.BCELoss()
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, gamma = 0.1, last_epoch=-1)


        for key in self.models.keys():
            for name, param in self.models[key].named_parameters():
                if param.requires_grad:
                    pass
                else:
                    print(name)
                    # print(param.data)
                    print("requires_grad:", param.requires_grad)
                    print("-----------------------------------")

        if self.local_rank == 0:
            self.log_print("Training model named: {}".format(self.opt.model_name))

        datasets_dict = {
            "ddad": datasets.DDADDataset,
            "nusc": datasets.NuscDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        self.opt.batch_size = self.opt.batch_size // self.opt.cam_N

        # pdb.set_trace()

        if self.opt.dataset == 'nusc':
            # pdb.set_trace()
            # nusc = NuScenes(version='v1.0-trainval', dataroot=osp.join(self.opt.dataroot, 'nuscenes'), verbose=False)
            nusc = NuScenes(version='v1.0-trainval', dataroot=osp.join(self.opt.dataroot), verbose=False)

        elif self.opt.dataset == 'ddad':
            nusc = None

        else:
            nusc = None
            
        # pdb.set_trace()
        train_dataset = self.dataset(self.opt,
                                     self.opt.height, self.opt.width,
                                     self.opt.frame_ids, num_scales=self.num_scales, is_train=True,
                                     volume_depth=self.opt.volume_depth, nusc=nusc)

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)

        # pdb.set_trace()
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, collate_fn=self.my_collate,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)

        self.num_total_steps = len(self.train_loader) * self.opt.num_epochs

        val_dataset = self.dataset(self.opt,
                                   self.opt.height, self.opt.width,
                                   self.opt.frame_ids, num_scales=1, is_train=False,
                                   volume_depth=self.opt.volume_depth, nusc=nusc)


        rank, world_size = get_dist_info()
        self.world_size = world_size
        val_sampler = DistributedSampler(val_dataset, world_size, rank, shuffle=False)


        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, collate_fn=self.my_collate,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False, sampler=val_sampler)


        self.num_val = len(val_dataset)

        self.opt.batch_size = self.opt.batch_size * self.opt.cam_N
        self.num_val = self.num_val * self.opt.cam_N

        self.best_result_str = ''
        self.best_abs_rel = 1.0

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            num_cam = self.opt.cam_N * 3 if self.opt.auxiliary_frame else self.opt.cam_N
            self.backproject_depth[scale] = BackprojectDepth(num_cam, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(num_cam, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        if self.local_rank == 0:
            self.log_print("There are {:d} training items and {:d} validation items\n".format(
                len(train_dataset), len(val_dataset)))
        
        if self.opt.use_semantic:
            if len(self.opt.class_frequencies) == self.opt.semantic_classes:
                self.class_weights = 1.0 / np.sqrt(np.array(self.opt.class_frequencies, dtype=np.float32))
                self.class_weights = np.nan_to_num(self.class_weights, posinf=0)
                self.class_weights = self.class_weights / np.mean(self.class_weights)
                self.sem_criterion = nn.CrossEntropyLoss(
                    weight=torch.FloatTensor(self.class_weights).to(self.device),
                    ignore_index=-1, reduction="mean")
            else:
                self.sem_criterion = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.save_opts()

        self.silog_criterion = silog_loss(variance_focus=0.85)


    def my_collate(self, batch):
        batch_new = {}
        keys_list = list(batch[0].keys())
        special_key_list = ['id', 'token', 'scene_name', 'frame_idx']

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

        special_key_list = ['id', 'token', ('K_ori', -1), ('K_ori', 1)]

        for key, ipt in inputs.items():

            if key in special_key_list:
                inputs[key] = ipt

            else:
                inputs[key] = ipt.to(self.device)


    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline"""
        if self.local_rank == 0:

            os.makedirs(osp.join(self.log_path, 'code'), exist_ok=True)

            # back up files
            source1 = 'runner.py'
            source3 = 'run.py'
            source4 = 'options.py'
            source5 = 'run_vis.py'
            source10 = 'run_gs_occ.sh'

            source6 = 'configs'
            source7 = 'networks'
            source8 = 'datasets'
            source9 = 'utils'
            
            source = [source1, source3, source4, source5, source10]
            
            for i in source:
                shutil.copy(i, osp.join(self.log_path, 'code'))

            if not osp.exists(osp.join(self.log_path, 'code' + '/configs')):
                shutil.copytree(source6, osp.join(self.log_path, 'code' + '/configs'))

            if not osp.exists(osp.join(self.log_path, 'code' + '/networks')):
                shutil.copytree(source7, osp.join(self.log_path, 'code' + '/networks'))

            if not osp.exists(osp.join(self.log_path, 'code' + '/datasets')):
                shutil.copytree(source8, osp.join(self.log_path, 'code' + '/datasets'))

            if not osp.exists(osp.join(self.log_path, 'code' + '/utils')):
                shutil.copytree(source9, osp.join(self.log_path, 'code' + '/utils'))

        self.step = 1

        if self.opt.eval_only:
            self.val(epoch = 'final')
            if self.local_rank == 0:
                self.evaluation(evl_score=True, evl_rayiou=self.opt.rayiou)

            return None

        self.epoch = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            print("========== NEW EPOCH : " + str(self.epoch) + "==========")

            self.train_loader.sampler.set_epoch(self.epoch)
            self.run_epoch()
            
            # self.save_model()

            self.val(epoch=self.epoch)

            if self.local_rank == 0:
                self.log_print(f"Evaluation results at epoch {self.epoch} (step {self.step}):")
                self.evaluation(evl_score=True)
                
        self.save_model()

        self.val(epoch=self.epoch)

        if self.local_rank == 0:
            self.log_print(f"Evaluation results at epoch {self.epoch} (step {self.step}):")
            self.evaluation(evl_score=True , evl_rayiou=self.opt.rayiou)

        return None

    def evaluation(self, evl_score=False, evl_rayiou = False, step=None):

        batch_size = self.world_size

        if self.local_rank == 0:
            self.log_print("-> Evaluating {} in {}".format('final', batch_size))

            errors = {}
            # if self.opt.self_supervise:
            eval_types = ['scale-aware']
            # else:
            #     eval_types = ['scale-ambiguous', 'scale-aware']
            
            for eval_type in eval_types:
                errors[eval_type] = {}

            for i in range(batch_size):
                while not osp.exists(osp.join(self.log_path, 'eval', '{}.pkl'.format(i))):
                    time.sleep(10)
                time.sleep(5)
                with open(osp.join(self.log_path, 'eval', '{}.pkl'.format(i)), 'rb') as f:
                    errors_i = pickle.load(f)
                    for eval_type in eval_types:
                        for camera_id in errors_i[eval_type].keys():
                            if camera_id not in errors[eval_type].keys():
                                errors[eval_type][camera_id] = []

                            errors[eval_type][camera_id].append(errors_i[eval_type][camera_id])

                    if self.opt.eval_occ and self.opt.use_semantic:
                        if i == 0:
                            errors['class_names'] = errors_i['class_names']
                            errors['mIoU'] = [errors_i['mIoU']]
                            errors['cnt'] = [errors_i['cnt']]
                        else:
                            errors['mIoU'].append(errors_i['mIoU'])
                            errors['cnt'].append(errors_i['cnt'])
                    elif self.opt.eval_occ:
                        if i == 0:
                            errors['acc'] = [errors_i['acc']]
                            errors['comp'] = [errors_i['comp']]
                            errors['f1'] = [errors_i['f1']]
                            errors['acc_dist'] = [errors_i['acc_dist']]
                            errors['cmpl_dist'] = [errors_i['cmpl_dist']]
                            errors['cd'] = [errors_i['cd']]
                            errors['cnt'] = [errors_i['cnt']]
                        else:
                            errors['acc'].append(errors_i['acc'])
                            errors['comp'].append(errors_i['comp'])
                            errors['f1'].append(errors_i['f1'])
                            errors['cnt'].append(errors_i['cnt'])

            if self.opt.eval_occ and self.opt.use_semantic:
                class_names = errors['class_names']
                mIoUs = np.stack(errors['mIoU'], axis=0)
                cnts = np.array(errors['cnt'])
                weights = cnts / np.sum(cnts)
                IoUs = np.sum(mIoUs * np.expand_dims(weights, axis=1), axis=0)
                index_without_others = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]  # without 0 and 12
                index_without_empty = [1, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16]  # without 0, 2, 6, 12
                mIoU_without_others = np.mean(IoUs[index_without_others])
                mIoU_without_empty = np.mean(IoUs[index_without_empty])
                self.log_print(f"Classes: {class_names}")
                self.log_print(f"IoUs: {IoUs}")
                self.log_print(f"mIoU without others: {mIoU_without_others}")
                self.log_print(f"mIoU without empty: {mIoU_without_empty}")
            
            
            elif self.opt.eval_occ:
                acc = np.array(errors['acc'])
                comp = np.array(errors['comp'])
                f1 = np.array(errors['f1'])
                acc_dist = np.array(errors['acc_dist'])
                cmpl_dist = np.array(errors['cmpl_dist'])
                cd = np.array(errors['cd'])
                cnts = np.array(errors['cnt'])
                weights = cnts / np.sum(cnts)
                acc_mean = np.sum(acc * weights)
                comp_mean = np.sum(comp * weights)
                f1_mean = np.sum(f1 * weights)
                acc_dist_mean = np.sum(acc_dist * weights)
                cmpl_dist_mean = np.sum(cmpl_dist * weights)
                cd_mean = np.sum(cd * weights)
                self.log_print(f"Precision: {acc_mean}")
                self.log_print(f"Recal: {comp_mean}")
                self.log_print(f"F1: {f1_mean}")
                self.log_print(f"Acc: {acc_dist_mean}")
                self.log_print(f"Comp: {cmpl_dist_mean}")
                self.log_print(f"CD: {cd_mean}")


            num_sum = 0
            for eval_type in eval_types:
                for camera_id in errors[eval_type].keys():
                    errors[eval_type][camera_id] = np.concatenate(errors[eval_type][camera_id], axis=0)

                    if eval_type == 'scale-aware':
                        num_sum += errors[eval_type][camera_id].shape[0]

                    errors[eval_type][camera_id] = np.nanmean(errors[eval_type][camera_id], axis=0)

            for eval_type in eval_types:
                self.log_print("{} evaluation:".format(eval_type))
                mean_errors_sum = 0
                for camera_id in errors[eval_type].keys():
                    mean_errors_sum += errors[eval_type][camera_id]
                mean_errors_sum /= len(errors[eval_type].keys())
                errors[eval_type]['all'] = mean_errors_sum

                for camera_id in errors[eval_type].keys():
                    mean_errors = errors[eval_type][camera_id]
                    self.log_print(camera_id)
                    self.log_print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                    self.log_print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()))

                if mean_errors_sum[0] < self.best_abs_rel:
                    self.best_abs_rel = mean_errors_sum[0]
                    self.best_result_str = ("&{: 8.3f}  " * 7).format(*mean_errors_sum.tolist())
                self.log_print("best result ({}):".format(eval_type))
                self.log_print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                self.log_print(self.best_result_str)

            assert num_sum == self.num_val

            if evl_rayiou == 'yes':
                # pdb.set_trace()
                str=('python ray_metrics.py --pred_dir {}'.format(osp.join(os.getcwd(), self.log_path, 'visual_feature', 'final'))) 
                # str=('python ray_metrics.py --pred_dir {}'.format('/home/wsgan/project/bev/OccNeRF/logs/0817_rerun_debug_all/seg_True_0.02_0.1_0.1_pose_No_yes_mask_False/type_prob_0.0_s_0.2_180_l_self_en_w_0.1_ep_12_f_False_infi_True_cont_True_depth_0.1_80.0/exp_2024_08_17_01_34_08/visual_feature/final')) 
                # os.chdir('/home/wsgan/project/bev/CVPR2024-Occupancy-Flow-Challenge/SparseOcc')
                os.system(str)


    def val(self, save_image=True, epoch = 'final'):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        errors = {}
        
        # if self.opt.self_supervise:
        eval_types = ['scale-aware']
        
        # else:
        #     eval_types = ['scale-ambiguous', 'scale-aware']
        
        for eval_type in eval_types:
            errors[eval_type] = {}

        self.models["encoder"].eval()
        self.models["depth"].eval()
        ratios_median = []

        print('begin eval!')
        total_time = []

        total_abs_rel_26 = []
        total_sq_rel_26 = []
        total_rmse_26 = []
        total_rmse_log_26 = []
        total_a1_26 = []
        total_a2_26 = []
        total_a3_26 = []

        # depth occupancy
        total_abs_rel_52 = []
        total_sq_rel_52 = []
        total_rmse_52 = []
        total_rmse_log_52 = []
        total_a1_52 = []
        total_a2_52 = []
        total_a3_52 = []
        
        if self.opt.use_semantic and self.opt.eval_occ:

            occ_eval_metrics = occ_metrics.Metric_mIoU(
                num_classes=18, use_lidar_mask=False, use_image_mask=True)
        
        
        elif self.opt.eval_occ:
            occ_eval_metrics = occ_metrics.Metric_FScore(use_image_mask=True)
        else:
            occ_eval_metrics = None

        total_evl_time = time.time()

        with torch.no_grad():

            loader = self.val_loader

            for idx, data in enumerate(loader):
                # continue
                # exit()
                eps_time = time.time()

                input_color = data[("color", 0, 0)][:self.opt.cam_N].cuda()

                gt_depths = data["depth"].cpu().numpy()
                camera_ids = data["id"]

                features = self.models["encoder"](input_color)

                if self.opt.use_t != 'No':
                    prev_inputs_color = data["color", -1, 0][:2*self.opt.cam_N].cuda()
                    prev_feature = self.models["encoder"](prev_inputs_color)
                    features = [torch.cat([features_i, prev_feature_i], dim=0) for features_i, prev_feature_i in zip(features, prev_feature)]

                output = self.models["depth"](features, data, epoch = 0, is_train=False)

                eps_time = time.time() - eps_time

                eps_time = eps_time # - output['render_time']

                # pdb.set_trace()
                
                total_time.append(eps_time)

                if self.opt.volume_depth and self.opt.eval_occ:

                    if self.opt.use_semantic:
                        # mIoU, class IoU
                        # pdb.set_trace()
                        semantics_pred = output['pred_occ_logits'][0].argmax(0)

                        occ_eval_metrics.add_batch(
                            semantics_pred=semantics_pred.detach().cpu().numpy(),
                            semantics_gt=data['semantics_3d'].detach().cpu().numpy(),
                            mask_camera=data['mask_camera_3d'].detach().cpu().numpy().astype(bool),
                            mask_lidar=None)
                        
                        if self.opt.rayiou == 'yes' and epoch == 'final':
                            # save npy
                            semantic_save_path = osp.join(self.log_path, 'visual_feature', epoch, data['token'][0] + '.npz')
                            os.makedirs(osp.join(self.log_path, 'visual_feature',epoch), exist_ok=True)
                            np.savez_compressed(semantic_save_path, pred=semantics_pred.to('cpu').numpy())


                        if self.local_rank == 0 and idx % 20 == 0:
                            _, miou, _ = occ_eval_metrics.count_miou()
                            print('mIoU:', miou)


                    else:
                        # Acc, Comp, Precision, Recall, Chamfer, F1
                        occ_prob = output['pred_occ_logits'][0, -1].sigmoid()

                        if self.opt.last_free:

                            occ_prob = 1.0 - occ_prob

                        free_mask = occ_prob < 0.6  # TODO: threshold

                        occ_pred = torch.zeros_like(free_mask, dtype=torch.long)

                        occ_pred[free_mask] = 17   

                        # pdb.set_trace()

                        occ_eval_metrics.add_batch(
                            semantics_pred=occ_pred.detach().cpu().numpy(),
                            semantics_gt=data['semantics_3d'].detach().cpu().numpy(),
                            mask_camera=data['mask_camera_3d'].detach().cpu().numpy().astype(bool),
                            mask_lidar=None)
                        

                        # pdb.set_trace()
                        if self.local_rank == 0 and idx % 20 == 0:
                            _, _, f1, _, _, cd, _ = occ_eval_metrics.count_fscore()
                            print('f1:', f1)
                            print('cd:', cd)


                if self.local_rank == 0 and idx % 100 == 0:
                    print('single inference:(eps time:', eps_time, 'secs)')

                if self.opt.volume_depth:
                    pred_disps_flip = output[("disp", 0)]


                pred_disps = pred_disps_flip.cpu()[:, 0].numpy()


                concated_image_list = []
                concated_depth_list = []

                for i in range(pred_disps.shape[0]):

                    camera_id = camera_ids[i]

                    if camera_id not in list(errors['scale-aware']):
                        errors['scale-aware'][camera_id] = []
                        if 'scale-ambiguous' in errors.keys():
                            errors['scale-ambiguous'][camera_id] = []

                    gt_depth = gt_depths[i]
                    gt_height, gt_width = gt_depth.shape[:2]

                    pred_disp = pred_disps[i]

                    if self.opt.volume_depth:
                        pred_depth = pred_disp

                        if self.local_rank == 0 and idx % 100 == 0 and i == 0:
                            print('volume rendering depth: min {}, max {}'.format(pred_depth.min(), pred_depth.max()))

                    pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))


                    mask = np.logical_and(gt_depth > self.opt.min_depth_test, gt_depth < self.opt.max_depth_test)

                    pred_depth_color = visualize_depth(pred_depth.copy())
                    color = (input_color[i].cpu().permute(1, 2, 0).numpy()) * 255
                    color = color[..., [2, 1, 0]]

                    concated_image_list.append(color)
                    concated_depth_list.append(cv2.resize(pred_depth_color.copy(), (self.opt.width, self.opt.height)))

                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    ratio_median = np.median(gt_depth) / np.median(pred_depth)
                    ratios_median.append(ratio_median)
                    pred_depth_median = pred_depth.copy() * ratio_median

                    if 'scale-ambiguous' in errors.keys():
                        pred_depth_median[pred_depth_median < self.opt.min_depth_test] = self.opt.min_depth_test
                        pred_depth_median[pred_depth_median > self.opt.max_depth_test] = self.opt.max_depth_test

                        errors['scale-ambiguous'][camera_id].append(compute_errors(gt_depth, pred_depth_median))

                    pred_depth[pred_depth < self.opt.min_depth_test] = self.opt.min_depth_test
                    pred_depth[pred_depth > self.opt.max_depth_test] = self.opt.max_depth_test

                    errors['scale-aware'][camera_id].append(compute_errors(gt_depth, pred_depth))


                save_frequency = self.opt.save_frequency


                if save_image and idx % save_frequency == 0 and self.local_rank == 0:
                    
                    print('idx:', idx)

                    if self.opt.cam_N == 6:
                        image_left_front_right = np.concatenate(
                            (concated_image_list[1], concated_image_list[0], concated_image_list[5]), axis=1)
                        image_left_rear_right = np.concatenate(
                            (concated_image_list[4], concated_image_list[3], concated_image_list[2]), axis=1)

                        image_surround_view = np.concatenate((image_left_front_right, image_left_rear_right), axis=0)

                        depth_left_front_right = np.concatenate(
                            (concated_depth_list[1], concated_depth_list[0], concated_depth_list[5]), axis=1)
                        depth_right_rear_left = np.concatenate(
                            (concated_depth_list[4], concated_depth_list[3], concated_depth_list[2]), axis=1)

                        depth_surround_view = np.concatenate((depth_left_front_right, depth_right_rear_left), axis=0)
                        surround_view = np.concatenate((image_surround_view, depth_surround_view), axis=0)
                    
                    elif self.opt.cam_N == 1:
                        surround_view = np.concatenate((concated_image_list[0], concated_depth_list[0]), axis=0)

                    # pdb.set_trace()
                    os.makedirs(osp.join(self.log_path, 'visual_rgb_depth', 'epoch_{}'.format(epoch)), exist_ok=True)
                    cv2.imwrite('{}/visual_rgb_depth/epoch_{}/{}-{}.jpg'.format(self.log_path, epoch, self.local_rank, idx), surround_view)

                    vis_dic = {}
                    vis_dic['opt'] = self.opt
                    # vis_dic['depth_color'] = concated_depth_list
                    # vis_dic['rgb'] = concated_image_list
                    vis_dic['pose_spatial'] = data['pose_spatial'].detach().cpu().numpy()
                    # vis_dic['probability'] = output['density'].detach().cpu().numpy()
                    # np.save('{}/visual_feature/{}-{}.npy'.format(self.log_path, self.local_rank, idx), vis_dic)


        for eval_type in eval_types:
            for camera_id in errors[eval_type].keys():
                errors[eval_type][camera_id] = np.array(errors[eval_type][camera_id])
        
        if self.opt.use_semantic and self.opt.eval_occ:
            class_names, mIoU, cnt = occ_eval_metrics.count_miou()
            errors['class_names'] = class_names
            errors['mIoU'] = mIoU
            errors['cnt'] = cnt

        elif self.opt.eval_occ:
            acc, comp, f1, acc_dist, cmpl_dist, cd, cnt = occ_eval_metrics.count_fscore()
            errors['acc'] = acc
            errors['comp'] = comp
            errors['f1'] = f1
            errors['acc_dist'] = acc_dist
            errors['cmpl_dist'] = cmpl_dist
            errors['cd'] = cd
            errors['cnt'] = cnt

        with open(osp.join(self.log_path, 'eval', '{}.pkl'.format(self.local_rank)), 'wb') as f:
            pickle.dump(errors, f)

        eps_time = time.time() - total_evl_time

        if self.local_rank == 0:
            self.log_print('median: {}'.format(np.array(ratios_median).mean()))
            self.log_print('mean inference time: {}'.format(np.array(total_time).mean()))
            self.log_print('total evl time: {} h'.format(eps_time / 3600))

        print('finish eval!')

        self.set_train()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        torch.autograd.set_detect_anomaly(True)
        if self.local_rank == 0:
            print("Training")
        self.set_train()

        if self.local_rank == 0:
            self.log_print_train('self.epoch: {}, lr: {}'.format(self.epoch, self.model_lr_scheduler.get_last_lr()))

        scaler = torch.cuda.amp.GradScaler(enabled=self.opt.use_fp16, init_scale=2**8)
        len_loader = len(self.train_loader)

        # pdb.set_trace()
        
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            scaler.scale(losses["loss"]).backward()
            scaler.step(self.model_optimizer)
            scaler.update()
            self.model_optimizer.zero_grad()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 200 == 0

            # pdb.set_trace()
            if early_phase or late_phase or (self.epoch == (self.opt.num_epochs - 1)):
                self.log_time(batch_idx, len_loader, duration, losses)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

            # if self.step > 0 and self.step % self.opt.eval_frequency == 0 and self.opt.eval_frequency > 0:
            #     # self.save_model()
            #     self.val()
            #     if self.local_rank == 0:
            #         self.evaluation()

            self.step += 1

        self.model_lr_scheduler.step()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        # pdb.set_trace()

        __p = lambda x: basic.pack_seqdim(x, self.opt.B)  # merge batch and number of cameras
        __u = lambda x: basic.unpack_seqdim(x, self.opt.B)

        self.to_device(inputs)
        
        with torch.cuda.amp.autocast(enabled=self.opt.use_fp16):

            # if use pass frame, inputs["color_aug", 0, 0][:self.opt.cam_N] + inputs["color_aug", -1, 0][:2*self.opt.cam_N]
            # 需要把梯度去掉节省内存

            inputs_color =  inputs["color_aug", 0, 0][:self.opt.cam_N]
            features = self.models["encoder"](inputs_color)


            if self.opt.use_t != 'No':
                
                with torch.no_grad():
                    prev_inputs_color = inputs["color_aug", -1, 0][:2*self.opt.cam_N]
                    prev_feature = self.models["encoder"](prev_inputs_color)

                feat_length = len(features)
                features = [torch.cat([features_i, prev_feature_i], dim=0) for features_i, prev_feature_i in zip(features, prev_feature)]


        features = [feat.float() for feat in features]
        
        outputs = self.models["depth"](features, inputs, self.epoch)
        # Note that for volume depth, outputs[("disp", 0)] is depth

        if self.opt.gt_pose != 'No':  # True

            with torch.no_grad():
                pose_dict = self.predict_poses(inputs, features)
            outputs.update(pose_dict)


        losses = {}
        losses['loss'] = 0

        if self.opt.render_type == 'gt':

            losses['loss'] = outputs[("loss_gt_occ", 0)]

            return outputs, losses

        

        if self.opt.self_supervise != 'gt': # true

            self.generate_images_pred(inputs, outputs)
            # pdb.set_trace()
            losses = self.compute_self_supervised_losses(inputs, outputs, losses)

            losses['loss'] += losses['self_loss']

        if self.opt.self_supervise != 'self':

            depth_gt = inputs['depth']
            mask = (depth_gt > self.opt.min_depth) & (depth_gt < self.opt.max_depth)
            mask.detach_()
            disp  = outputs[('disp', 0)][:6,...]
            gt_loss = self.opt.gtw * self.get_gt_loss(disp, depth_gt, mask)

            losses['loss_gt'] = gt_loss
            losses['loss'] += gt_loss


        return outputs, losses

    
    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:

            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

            coarse_RT = None

            axisangle, translation = self.models["pose"](pose_inputs, joint_pose=True, coarse_RT = coarse_RT)
            
            if self.opt.detach_pose != 'No':
                axisangle = axisangle.detach()
                translation = translation.detach()

            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            trans = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
            trans = trans.unsqueeze(1).repeat(1, 6, 1, 1).reshape(-1, 4, 4) # torch.Size([6, 4, 4]

            # pdb.set_trace()
            outputs[("cam_T_cam", 0, f_i)] = torch.linalg.inv(inputs["pose_spatial"]) @ trans @ inputs["pose_spatial"]

        return outputs


    def get_gt_loss(self, disp, depth_gt, mask):
        

        singel_scale_total_loss = 0

        if self.opt.volume_depth:


            depth_pred = disp

            depth_pred = F.interpolate(depth_pred, size=[self.opt.height_ori, self.opt.width_ori], mode="bilinear", align_corners=False).squeeze(1)

            # pdb.set_trace()
            no_aug_loss = self.silog_criterion.forward(depth_pred, depth_gt, mask.to(torch.bool), self.opt)

            singel_scale_total_loss += no_aug_loss



   
        return singel_scale_total_loss


    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                if scale == 0:
                    outputs[("disp", scale)] = disp
                source_scale = 0

            if self.opt.volume_depth:
                depth = disp
            else:
                depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth, abs=False)
            
            outputs[("depth", 0, scale)] = depth


            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if self.opt.gt_pose != 'No':
                    T =  outputs[("cam_T_cam", 0, frame_id)]
                
                else:
                    T = inputs[("cam_T_cam", frame_id)]

                
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", 0, source_scale)])
                
                
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", frame_id, source_scale)], T)

                
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border", align_corners=True)

                
                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]
    

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    def compute_self_supervised_losses(self, inputs, outputs,  losses = {}):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        # losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            if self.opt.use_fix_mask:
                output_mask = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            # pdb.set_trace()
            disp = outputs[("disp", scale)]

            # print('Scale {},  disp min {}, max {}'.format(scale, disp[0].min(), disp[0].max()))
            min_depth = disp[0].min()
            max_depth = disp[0].max()


            if self.opt.volume_depth:  # in fact, it is depth
                disp = 1.0 / (disp + 1e-7)

            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]  
            # 多帧监督的时候的target frame 是 【-1， 0， 1】
            
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.use_fix_mask:
                reprojection_losses *= inputs["mask"] #* output_mask

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            # pdb.set_trace()
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            losses[f"loss_pe/{scale}"] = loss

            semantic_loss = 0.0

            if self.opt.use_semantic and scale == 0:
                # pdb.set_trace()
                pred_semantic = outputs[("semantic", 0)].float()
                target_semantic = inputs["semantic"]
                

                # target_semantic[target_semantic > 0] = target_semantic[target_semantic > 0] - 1
                target_semantic[target_semantic > 0] = target_semantic[target_semantic > 0]
                
                target_semantic = F.interpolate(target_semantic.unsqueeze(1).float(), size=pred_semantic.shape[1:3], mode="nearest").squeeze(1)
                
                # pdb.set_trace()
                if self.opt.use_fix_mask:
                    ddad_mask = F.interpolate(inputs["mask"][:,:1, :, :], size=pred_semantic.shape[1:3], mode="nearest").squeeze(1)
                    ddad_mask = ddad_mask.bool()
                    semantic_loss += self.sem_criterion(pred_semantic[ddad_mask], target_semantic[ddad_mask].long())
                
                else:
                    semantic_loss += self.sem_criterion(pred_semantic.view(-1, self.opt.semantic_classes), target_semantic.view(-1).long())
                
                semantic_loss = self.opt.semantic_loss_weight * semantic_loss
                losses[f"loss_semantic/{scale}"] = semantic_loss
            

            loss_reg = 0
            for k, v in outputs.items():
                if isinstance(k, tuple) and k[0].startswith("loss") and k[1] == scale:
                    losses[f"{k[0]}/{k[1]}"] = v
                    loss_reg += v
            
            total_loss += loss + loss_reg + semantic_loss
            # losses["loss/{}".format(scale)] = loss + loss_reg + semantic_loss

        total_loss /= self.num_scales

        losses["self_loss"] = total_loss

        losses["min_d"] = min_depth
        losses["max_d"] = max_depth

        return losses


    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0
        _, _, H, W = depth_gt.shape

        depth_pred = outputs[("depth", 0, 0)].detach()
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [H, W], mode="bilinear", align_corners=False), 1e-3, self.opt.max_depth)

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        if 'cam_T_cam' not in inputs:
            depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=self.opt.max_depth)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())


    def log_time(self, batch_idx, len_loader, duration, loss_dict):
        """Print a logging statement to the terminal
        """
        if self.local_rank == 0:
            samples_per_sec = self.opt.batch_size / duration
            time_sofar = time.time() - self.start_time
            training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0

            loss_info = ''
            for l, v in loss_dict.items():
                loss_info += "{}: {:.4f} | ".format(l, v)
            print_string = "epoch {:>2}/{:>2} | batch {:>5}/{:>5} | examples/s: {:3.1f}" + \
                           " | {}time elapsed: {} | time left: {}"

            self.log_print_train(print_string.format(self.epoch+1, self.opt.num_epochs, batch_idx+1, len_loader, samples_per_sec, loss_info,
                                               sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))


    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                writer.add_image(
                    "disp_{}/{}".format(s, j),
                    normalize_image(outputs[("disp", s)][j]), self.step)

                if self.opt.predictive_mask:
                    for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
                        writer.add_image(
                            "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step)

                elif not self.opt.disable_automasking:
                    writer.add_image(
                        "automask_{}/{}".format(s, j),
                        outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

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

    def save_model(self):
        """Save model weights to disk
        """
        if self.local_rank == 0:
            save_folder = osp.join(self.log_path, "models", "weights_{}".format(self.step))
            if not osp.exists(save_folder):
                os.makedirs(save_folder)

            for model_name, model in self.models.items():
                save_path = osp.join(save_folder, "{}.pth".format(model_name))
                to_save = model.module.state_dict()
                if model_name == 'encoder':
                    # save the sizes - these are needed at prediction time
                    to_save['height'] = self.opt.height
                    to_save['width'] = self.opt.width
                torch.save(to_save, save_path)

            save_path = osp.join(save_folder, "{}.pth".format("adam"))
            torch.save(self.model_optimizer.state_dict(), save_path)

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

    def load_optimizer(self):
        # loading adam state
        optimizer_load_path = osp.join(self.opt.load_weights_folder, "adam.pth")
        if osp.isfile(optimizer_load_path):
            if self.local_rank == 0:
                self.log_print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            self.log_print("Cannot find Adam weights so Adam is randomly initialized")

    def log_print(self, str):
        print(str)
        with open(osp.join(self.log_path, 'log.txt'), 'a') as f:
            f.writelines(str + '\n')

    def log_print_train(self, str):
        print(str)
        with open(osp.join(self.log_path, 'log_train.txt'), 'a') as f:
            f.writelines(str + '\n')


if __name__ == "__main__":
    pass