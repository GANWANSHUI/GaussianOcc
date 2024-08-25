import os
import glob
import mmcv
import argparse
import numpy as np
import torch, pdb
from torch.utils.data import DataLoader

from utils.rayiou_metric import main_rayiou
from datasets.ego_pose_dataset import EgoPoseDataset


occ3d_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free']

openocc_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free']


def main(args):
    data_infos = mmcv.load(os.path.join(args.data_root, 'nuscenes_infos_val.pkl'))['infos']
    gt_filepaths = sorted(glob.glob(os.path.join(args.data_root, args.data_type, '*/*/*.npz')))

    gt_path_root = 'data/nuscenes/gts'
    gt_filepaths = sorted(glob.glob(os.path.join(gt_path_root, '*/*/*.npz')))


    # retrieve scene_name
    token2scene = {}
    for gt_path in gt_filepaths:
        token = gt_path.split('/')[-2]
        scene_name = gt_path.split('/')[-3]
        token2scene[token] = scene_name

    for i in range(len(data_infos)):
        
        scene_name = token2scene[data_infos[i]['token']]
        data_infos[i]['scene_name'] = scene_name

    lidar_origins = []
    occ_gts = []
    occ_preds = []

    # pdb.set_trace()

    # data_infos = data_infos[:50]

    for idx, batch in enumerate(DataLoader(EgoPoseDataset(data_infos), num_workers=8)):
        output_origin = batch[1]
        info = data_infos[idx]
        occ_path = os.path.join(gt_path_root, info['scene_name'], info['token'], 'labels.npz')
        occ_gt = np.load(occ_path, allow_pickle=True)['semantics']
        occ_gt = np.reshape(occ_gt, [200, 200, 16]).astype(np.uint8)

        occ_path = os.path.join(args.pred_dir, info['token'] + '.npz')
        occ_pred = np.load(occ_path, allow_pickle=True)['pred']
        occ_pred = np.reshape(occ_pred, [200, 200, 16]).astype(np.uint8)

        lidar_origins.append(output_origin)
        occ_gts.append(occ_gt)
        occ_preds.append(occ_pred)

    if args.data_type == 'occ3d':
        occ_class_names = occ3d_class_names
    elif args.data_type == 'openocc_v2':
        occ_class_names = openocc_class_names
    else:
        raise ValueError
    
    result, table = main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)

    # pdb.set_trace()

    # log_path = args.pred_dir
    log_path = os.path.split(os.path.split(args.pred_dir)[0])[0]

    with open(os.path.join(log_path, 'log.txt'), 'a') as f:
        f.writelines(str(result) + '\n')
        # pdb.set_trace()
        table = f'metric table:{table}'
        f.writelines(table + '\n')
        
    # print(result)
    return (result, table)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default='data/nuscenes')
    parser.add_argument("--pred_dir", type=str, default= '')
    parser.add_argument("--data-type", type=str, choices=['occ3d', 'openocc_v2'], default='occ3d')
    args = parser.parse_args()

    torch.random.manual_seed(0)
    np.random.seed(0)

    main(args)