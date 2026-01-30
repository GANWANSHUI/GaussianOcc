import pickle
from copy import deepcopy

import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


dataroot = 'data/nuscenes'
nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
splits = create_splits_scenes()
val_scenes = splits['val']  # Define your own split.


print('Number of scenes in val split:', len(val_scenes))

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


cam_types = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
vis_data_dict = {'frames': []}
len_scenes = len(nusc.scene)
pbar = tqdm(total=len_scenes)
for i in range(len_scenes):
    scene = nusc.scene[i]

    if scene['name'] in val_scenes:
        scene_name = scene['name']
        print('proceing scene:', scene_name)
        first_sample_token = scene['first_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        sample_data = deepcopy(first_sample['data'])
        frame_idx = 0
        while True:
            frame_data = {
                'scene_name': scene_name,
                'frame_idx': frame_idx,
            }
            final = False
            for cam_type in cam_types:
                cam_token = sample_data[cam_type]
                cam = nusc.get('sample_data', cam_token)
                cam_calib = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                cam2ego = rt2mat(cam_calib['translation'], cam_calib['rotation'])
                intrinsics = cam_calib['camera_intrinsic']
                cam_data = {
                    'filename': cam['filename'],
                    'cam2ego': cam2ego,
                    'intrinsics': intrinsics,
                }
                next = cam['next']
                sample_data[cam_type] = next
                frame_data[cam_type] = cam_data
                if next == '':
                    final = True
                    break
            if final:
                break
            vis_data_dict['frames'].append(frame_data)
            frame_idx += 1
        # pbar.write(f'{scene_name}: {frame_idx}')
        pbar.update(1)
pbar.close()

with open('data/nuscenes/nuscenes_infos_vis_occnerf.pkl', 'wb') as f:
    pickle.dump(vis_data_dict, f)
