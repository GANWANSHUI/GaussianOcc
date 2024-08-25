#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import math, pdb
import numpy as np

# no semantic 
import diff_gaussian_rasterization as diff_3d_gaussian_rasterization
# semantic 
import diff_gaussian_rasterization_semantic as diff_3d_gaussian_rasterization_semantic


from simple_knn._C import distCUDA2
from .sh_utils import eval_sh
from .point_utils import depth_to_normal


def DistCUDA2(fused_point_cloud):

    dist2 = torch.clamp_min(distCUDA2(fused_point_cloud)[0], 0.0000001)
    scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

    return scales


def camera_intrinsic_fov(intrinsic):

    #计算FOV
    w, h = intrinsic[0][2]*2, intrinsic[1][2]*2
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    
    # Go
    fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_y = np.rad2deg(2 * np.arctan2(h, 2 * fy))

    return fov_x, fov_y


def splatting_render(viewpoint_camera, pc, scaling_modifier = 1.0,
           override_color = None, white_bg = False, opt = None):
    
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc['get_xyz'], dtype=pc['get_xyz'].dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
        
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera['FoVx'] * 0.5)
    tanfovy = math.tan(viewpoint_camera['FoVx'] * 0.5)

    #if min(pc.bg_color.shape) != 0:
    bg_color = torch.tensor([0., 0., 0.]).cuda()

    confidence = pc['confidence']
    
    # pdb.set_trace()
    if not opt.use_semantic:
        if opt.render_type == '3dgs':
            raster_settings = diff_3d_gaussian_rasterization.GaussianRasterizationSettings(
                image_height=int(viewpoint_camera['image_height']),
                image_width=int(viewpoint_camera['image_width']),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
                scale_modifier=scaling_modifier,

                viewmatrix=viewpoint_camera['world_view_transform'], # 外参
                projmatrix=viewpoint_camera['full_proj_transform'], # 3D -> 2D 

                sh_degree=pc['active_sh_degree'],
                campos=viewpoint_camera['camera_center'],
                prefiltered=False,
                debug=False, # False
                confidence=confidence)

            rasterizer = diff_3d_gaussian_rasterization.GaussianRasterizer(raster_settings=raster_settings)

    else:
        if opt.render_type == '3dgs':
            raster_settings = diff_3d_gaussian_rasterization_semantic.GaussianRasterizationSettings(
                image_height=int(viewpoint_camera['image_height']),
                image_width=int(viewpoint_camera['image_width']),
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg = bg_color, #torch.tensor([1., 1., 1.]).cuda() if white_bg else torch.tensor([0., 0., 0.]).cuda(), #bg_color,
                scale_modifier=scaling_modifier,

                viewmatrix=viewpoint_camera['world_view_transform'], # 外参
                projmatrix=viewpoint_camera['full_proj_transform'], # 3D -> 2D 

                sh_degree=pc['active_sh_degree'],
                campos=viewpoint_camera['camera_center'],
                prefiltered=False,
                debug=False,
                confidence = confidence)
            
            rasterizer = diff_3d_gaussian_rasterization_semantic.GaussianRasterizer(raster_settings=raster_settings)


    means3D = pc['get_xyz']
    means2D = screenspace_points
    opacity = pc['get_opacity']

    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc['get_scaling']
    rotations = pc['get_rotation']

    shs = None
    colors_precomp = None

    if 'semantic' in pc.keys():
        colors_precomp = pc['semantic']
    else:
        colors_precomp = None


    render_normal = None 
    render_dist = None 
    surf_depth = None 
    surf_normal = None 

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    # rasterizer return: color, feature_map, radii, depth, alpha
    # pdb.set_trace()

    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "depth": depth,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': depth,
            'surf_normal': surf_normal}


if __name__ == "__main__":

    pass