import os
import argparse
import copy

import numpy as np
import torch
from PIL import Image

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

# segment anything
from segment_anything import build_sam, SamPredictor
import numpy as np
import matplotlib.pyplot as plt

# diffusers
import torch
from huggingface_hub import hf_hub_download
# from nuscenes.nuscenes import NuScenes
import pickle
import concurrent.futures as futures
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


BOX_TRESHOLD = 0.20
TEXT_TRESHOLD = 0.20

TEXT_PROMPT1 = "Pedestrian, Bus, Truck"
TEXT_PROMPT2 = "Bicycle, Bicyclist"
TEXT_PROMPT3 = "Motorcycle, Motorcyclist"
TEXT_PROMPT4 = "Barrier, Barricade"
TEXT_PROMPT5 = "Sedan, Highway, Terrain, Tree, Sidewalk, Building, Bridge, Pole, Billboard, Light, Ashbin, Crane, Trailer, Cone, Sky"

dic = {'sedan': 1, 
       'highway': 2, 
       'bus': 3, 
       'truck': 4, 
       'terrain': 5, 
       'tree': 6, 
       'sidewalk': 7,
       'bicycle': 8, # 'bicycle': 8, 'bicyclist': 8,
       'bicyclist': 8, 
       'barrier': 9,
       'barricade': 9, 
       'person': 10,
       'pedestrian': 10,
       'building': 11, # manmade
       'bridge': 11, # manmade
       'pole': 11, # manmade
       'billboard': 11, # manmade
       'light': 11, # manmade
       'ashbin': 11, # manmade
       'motorcycle': 12,
       'motorcyclist': 12,
       'crane': 13, 
       'trailer': 14, 
       'cone': 15, 
       'sky': 16}

colors = np.array(
    [
        [0, 0, 0, 153],
        [255, 120, 50, 153],  # car              orange
        [255, 192, 203, 153],  # road              pink
        [255, 255, 0, 153],  # bus                  yellow
        [0, 150, 245, 153],  # truck                  blue
        [0, 255, 255, 153],  # terrain               cyan
        [0, 175, 0, 153],  # tree                 green
        [255, 0, 0, 153],  # sidewalk              red
        [255, 240, 150, 153],  # bycicle         light yellow
        [135, 60, 0, 153],  # barrier              brown
        [160, 32, 240, 153],  # person                purple
        [255, 0, 255, 153],  # building, light          dark pink
        # [175,   0,  75, 153],       # other_flat           dark red
        [139, 137, 137, 153], # motocycle         dark red
        [75, 0, 75, 153],  # crane                dard purple
        [150, 240, 80, 153],  # trailer              light green
        [230, 230, 250, 153],  # traffic_cone              white
        [200, 180, 0, 153],  # sky           dark orange
        [0, 255, 127, 153],  # ego car              dark cyan
        [255, 99, 71, 153], 
        [111, 111, 222, 153] # others
        # [0, 191, 255, 153] # others
    ]
).astype(np.uint8)

def load_model_hf(device='cpu'):
    cache_config_file = './models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/GroundingDINO_SwinB.cfg.py'
    cache_file = './models--ShilongLiu--GroundingDINO/snapshots/a94c9b567a2a374598f05c584e96798a170c56fb/groundingdino_swinb_cogcoor.pth'
    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)

    args.device = device

    checkpoint = torch.load(cache_file, map_location='cpu')

    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

def show_mask(mask, image, phrases, logits, random_color=True, visual=False):
    h, w = mask.shape[-2:]
    
    num_mask = mask.shape[0]
    
    logits_masks = torch.zeros((h, w, 1))
    for i in range(num_mask):
        m = mask[i]
        logits_mask = (m.cpu().reshape(h, w, 1) * logits[i].cpu().reshape(1, 1, -1))
        logits_masks = torch.concat([logits_masks, logits_mask], dim=-1)
    
    idx_mask = logits_masks.argmax(dim=-1).numpy()
    phrases.insert(0, 'uncertain')
    catagory_mask = np.vectorize(lambda x: phrases[x])(idx_mask)
    
    vectorized_mapping = np.vectorize(lambda x: dic.get(x, -1))
    norm_idx_mask = vectorized_mapping(catagory_mask).squeeze()
    mask_image = np.zeros((h, w, 4))
    
    # visual:
    if visual:
        for i in range(norm_idx_mask.shape[0]):
            for j in range(norm_idx_mask.shape[1]):
                mask_image[i, j] = colors[norm_idx_mask[i, j]]
        mask_image_pil = Image.fromarray((mask_image).astype(np.uint8)).convert("RGBA")
        img = Image.fromarray(image).convert("RGBA")
        img = Image.alpha_composite(img, mask_image_pil)
        return np.array(img), norm_idx_mask
    else:
        return norm_idx_mask

def show_mask_sam(mask, image, phrases, logits, random_color=True, visual=False):
    h, w = mask.shape[-2:]
    idx_mask = mask.argmax(dim=0).squeeze().cpu().numpy()
    catagory_mask = np.vectorize(lambda x: phrases[x])(idx_mask)
    
    vectorized_mapping = np.vectorize(lambda x: dic.get(x, -1))
    norm_idx_mask = vectorized_mapping(catagory_mask).squeeze()
    mask_image = np.zeros((h, w, 4))
    
    if visual:
        for i in range(norm_idx_mask.shape[0]):
            for j in range(norm_idx_mask.shape[1]):
                mask_image[i, j] = colors[norm_idx_mask[i, j]]
        mask_image_pil = Image.fromarray((mask_image).astype(np.uint8)).convert("RGBA")
        img = Image.fromarray(image).convert("RGBA")
        img = Image.alpha_composite(img, mask_image_pil)
        return np.array(img), norm_idx_mask
    else:
        return norm_idx_mask

def generate_semantic(groundingdino_model, sam_predictor, local_rank, image_filename=None, device='cpu', save_name=None):
    image_source, image = load_image(image_filename)
    image = image.to(device)

    prompt_list = TEXT_PROMPT1.split(",")
    boxes_list, logits_list, phrases_list = [], [], []
    for i in range(len(prompt_list)):
        text = prompt_list[i].strip()
        boxes, logits, phrases = predict(
        model=groundingdino_model.module, 
        image=image, 
        caption=text, 
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
        device='cuda')

        boxes_list.append(boxes)
        logits_list.append(logits)
        phrases_list += phrases

    prompts = [TEXT_PROMPT2, TEXT_PROMPT3]
    for PROMPT in prompts:
        boxes, logits, phrases = predict(
        model=groundingdino_model.module, 
        image=image, 
        caption=PROMPT,
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device='cuda')

                            
        boxes_list.append(boxes)
        logits_list.append(logits)
        phrases_list += phrases

    prompts2 = [TEXT_PROMPT4, TEXT_PROMPT5]
    for PROMPT in prompts2:
        boxes, logits, phrases = predict(
        model=groundingdino_model.module, 
        image=image, 
        caption=PROMPT,
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device='cuda')
        
        boxes_list.append(boxes)
        logits_list.append(logits)
        phrases_list += phrases

    boxes = torch.cat(boxes_list, dim=0)
    logits = torch.cat(logits_list, dim=0)
    phrases = phrases_list

    valid_indices = [index for index in range(len(phrases)) if phrases[index] in dic]

    boxes = boxes[valid_indices]
    logits = logits[valid_indices]
    phrases = [phrases[index] for index in valid_indices]

    sam_predictor.set_image(image_source)
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    
    if transformed_boxes.shape[0] == 0 or transformed_boxes == None:
        print("No box detected!")
        return np.ones((H, W)) * (-1)
    
    masks, logits_masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                return_logits=True,
                multimask_output = False,
            )
    
    phrases_sam = copy.deepcopy(phrases)
    
    visual = True # set true to save the visualized mask
    sam = True
    if visual:
        frame_with_mask, mask_np = show_mask(masks, image_source, phrases, logits, visual=visual)
        Image.fromarray(frame_with_mask).save('./tmp/{}_mask.png'.format(save_name))
        if sam:
            frame_with_mask_sam, mask_np_sam = show_mask_sam(logits_masks, image_source, phrases_sam, logits, visual=visual)
            Image.fromarray(frame_with_mask_sam).save('./tmp/{}_mask_sam.png'.format(save_name))
    else:
        mask_np = show_mask(masks, image_source, phrases, logits, visual=visual)
        if sam:    
            mask_np_sam = show_mask_sam(logits_masks, image_source, phrases_sam, logits, visual=visual)
    
    if sam:
        return mask_np, mask_np_sam
    else:
        return mask_np


from options import MonodepthOptions
from datasets.ddad_dataset import DDADDataset

class NuscenesDataset(DDADDataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", default=0, type=int)
    # args = parser.parse_args()
    
    # for DDAD parameters
    options = MonodepthOptions()
    mono_opts = options.parse()

    sam = False
    with torch.no_grad():
        torch.cuda.set_device(mono_opts.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device("cuda", mono_opts.local_rank)

        data_path = './data/ddad'
        # version = 'v1.0-trainval'
        split = 'train'

        # nusc = NuScenes(version=version, dataroot=data_path, verbose=False)
        # with open(f'./datasets/ddad/info_{split}.pkl', 'rb') as f:
        #     ddad_data = pickle.load(f)
        
        ddad_dataset = DDADDataset(mono_opts,
                        mono_opts.height, mono_opts.width,
                        mono_opts.frame_ids, num_scales=1, is_train=True,
                        volume_depth=mono_opts.volume_depth)

        train_sampler = DistributedSampler(ddad_dataset, shuffle=False)

        trainloader = DataLoader(ddad_dataset, batch_size=1, num_workers=64, sampler=train_sampler)
        
        groundingdino_model = load_model_hf()
        groundingdino_model = groundingdino_model.to(device)

        print('local_rank:', mono_opts.local_rank)

        groundingdino_model = DDP(groundingdino_model, device_ids=[mono_opts.local_rank], output_device=mono_opts.local_rank)
        
        sam_checkpoint = './sam_vit_h_4b8939.pth'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam = sam.to(device)
        
        sam = DDP(sam, device_ids=[mono_opts.local_rank], output_device=mono_opts.local_rank)

        sam_predictor = SamPredictor(sam, ddp=True)
        
        save_path = './data/ddad/ddad_semantic'

        if sam:
            save_path_sam = './data/ddad/ddad_semantic_sam'

        camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
        

        print('generating DDAD semantic using groundingdino and sam')

        from tqdm import tqdm
        for index_data in tqdm(trainloader):
            # load image
            for image_filepath in index_data['color_image_filepath']:
                name_list = image_filepath[0][:-4].split('/')
                # example: 
                # ['', 'data', 'ggeoinfo', 'Wanshui_BEV', 'data', 'ddad', 'raw_data', '000018', 'rgb', 'CAMERA_01', '15638057062802582']
                image_name = name_list[-1]

                if False:
                    pass

                else:
                    cam_number = name_list[-2]
                    scene_number = name_list[-4]

                    if sam:
                        mask, mask_sam = generate_semantic(groundingdino_model, sam_predictor, mono_opts.local_rank, image_filename=image_filepath[0], device=device, save_name=image_name)
                    else:
                        mask = generate_semantic(groundingdino_model, sam_predictor, mono_opts.local_rank, image_filename=image_filepath[0], device=device, save_name=image_name)
                    
                    mask = mask.astype(np.int8) # 900, 1600

                    os.makedirs(os.path.join(save_path, scene_number, cam_number), exist_ok=True)
                    mask.tofile(os.path.join(save_path, scene_number, cam_number, image_name + '_mask.bin'))

                    if sam:
                        mask_sam = mask_sam.astype(np.int8)
                        os.makedirs(os.path.join(save_path_sam, scene_number, cam_number), exist_ok=True)
                        mask_sam.tofile(os.path.join(save_path_sam, scene_number, cam_number, image_name + '_mask.bin'))
                    # Code for load: mask = np.fromfile(os.path.join('data/nuscenes/nuscenes_semantic', camera_sample['filename'][:-4] + '_mask.bin'), dtype=np.int8).reshape(900, 1600)
                
                    print('finish processing index = {}'.format(index_data['color_image_filepath'][0][0][:-4].split('/')[-1]))
            
            # print('finish processing index = {}'.format(index_data['color_image_filepath'][0][0][:-4].split('/')[-1]))
            
    