import pdb

import torch
import cv2
import numpy as np

from utils import basic

# import basic


# 为了gs加的函数：
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    
    return np.float32(Rt)

def setup_opengl_proj(w, h, k, c2w, near=0.01, far=100):

    viewpoint_camera = {}

    # world_view_transform
    W2C = np.linalg.inv(c2w)

    R = np.transpose(W2C[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = W2C[:3, 3]
    world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
    camera_center = world_view_transform.inverse()[3, :3]

    # projection_matrix
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    
    FoVx = 2 * np.arctan2(w, 2 * fx)
    FoVy = 2 * np.arctan2(h, 2 * fy)

    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(opengl_proj)).squeeze(0)

    
    viewpoint_camera['FoVx'] = FoVx
    viewpoint_camera['FoVy'] = FoVy
    viewpoint_camera['world_view_transform'] = world_view_transform
    viewpoint_camera['full_proj_transform'] = full_proj_transform
    viewpoint_camera['camera_center'] = camera_center
    viewpoint_camera['image_height'] = h
    viewpoint_camera['image_width'] = w

    return viewpoint_camera


    # splattam
    # tanfovx=w / (2 * fx)
    # tanfovy=h / (2 * fy)

    # sparseGS
    # k = k.to('cpu').numpy()
    # fx, fy = k[0][0], k[1][1]

    
    # tanfovx = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    # tanfovy = np.rad2deg(2 * np.arctan2(h, 2 * fy))

   

    # pdb.set_trace()
    # 2*math.atan(w/(2*fx))
    # 2*math.atan(h/(2*fy))

    # def focal2fov(focal, pixels):
    #     return 2*math.atan(pixels/(2*focal))
    

    # return opengl_proj, tanfovx, tanfovy
    




def eye_4x4(B, device='cuda'):
    rt = torch.eye(4, device=torch.device(device)).view(1,4,4).repeat([B, 1, 1])
    return rt

def safe_inverse(a): #parallel version
    B, _, _ = list(a.shape)
    inv = a.clone()
    r_transpose = a[:, :3, :3].transpose(1,2) #inverse of rotation matrix

    inv[:, :3, :3] = r_transpose
    inv[:, :3, 3:4] = -torch.matmul(r_transpose, a[:, :3, 3:4])

    return inv

def safe_inverse_single(a):
    r, t = split_rt_single(a)
    t = t.view(3,1)
    r_transpose = r.t()
    inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
    bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
    # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4)
    inv = torch.cat([inv, bottom_row], 0)
    return inv

def apply_4x4(RT, xyz):
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:,:,:3]
    return xyz2

def get_camM_T_camXs(origin_T_camXs, ind=0):
    B, S = list(origin_T_camXs.shape)[0:2]
    camM_T_camXs = torch.zeros_like(origin_T_camXs)
    for b in list(range(B)):
        camM_T_origin = safe_inverse_single(origin_T_camXs[b,ind])
        for s in list(range(S)):
            camM_T_camXs[b,s] = torch.matmul(camM_T_origin, origin_T_camXs[b,s])
    return camM_T_camXs

def split_rt_single(rt):
    r = rt[:3, :3]
    t = rt[:3, 3].view(3)
    return r, t

def split_rt(rt):
    r = rt[:, :3, :3]
    t = rt[:, :3, 3].view(-1, 3)
    return r, t

def merge_rt(r, t):
    # r is B x 3 x 3
    # t is B x 3
    B, C, D = list(r.shape)
    B2, D2 = list(t.shape)
    assert(C==3)
    assert(D==3)
    assert(B==B2)
    assert(D2==3)
    t = t.view(B, 3)
    rt = eye_4x4(B, device=t.device)
    rt[:,:3,:3] = r
    rt[:,:3,3] = t
    return rt


def xyd2pointcloud(xyd, pix_T_cam):
    # xyd is like a pointcloud but in pixel coordinates;
    # this means xy comes from a meshgrid with bounds H, W,
    # and d comes from a depth map
    B, N, C = list(xyd.shape)
    assert (C == 3)
    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    xyz = pixels2camera(xyd[:, :, 0], xyd[:, :, 1], xyd[:, :, 2], fx, fy, x0, y0)
    return xyz


def pixels2camera(x, y, z, fx, fy, x0, y0):
    # x and y are locations in pixel coordinates, z is a depth in meters
    # they can be images or pointclouds
    # fx, fy, x0, y0 are camera intrinsics
    # returns xyz, sized B x N x 3

    B = x.shape[0]

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])

    x = torch.reshape(x, [B, -1])
    y = torch.reshape(y, [B, -1])
    z = torch.reshape(z, [B, -1])

    # unproject
    x = (z / fx) * (x - x0)
    y = (z / fy) * (y - y0)

    xyz = torch.stack([x, y, z], dim=2)
    # B x N x 3
    return xyz


def camera2pixels(xyz, pix_T_cam):
    # xyz is shaped B x H*W x 3
    # returns xy, shaped B x H*W x 2

    fx, fy, x0, y0 = split_intrinsics(pix_T_cam)
    x, y, z = torch.unbind(xyz, dim=-1)
    B = list(z.shape)[0]

    fx = torch.reshape(fx, [B, 1])
    fy = torch.reshape(fy, [B, 1])
    x0 = torch.reshape(x0, [B, 1])
    y0 = torch.reshape(y0, [B, 1])
    x = torch.reshape(x, [B, -1])
    y = torch.reshape(y, [B, -1])
    z = torch.reshape(z, [B, -1])

    EPS = 1e-4
    z = torch.clamp(z, min=EPS)
    x = (x * fx) / z + x0
    y = (y * fy) / z + y0
    xy = torch.stack([x, y], dim=-1)
    return xy


def scale_intrinsics(K, sx, sy):
    fx, fy, x0, y0 = split_intrinsics(K)
    fx = fx*sx
    fy = fy*sy
    x0 = x0*sx
    y0 = y0*sy
    K = merge_intrinsics(fx, fy, x0, y0)
    return K

def split_intrinsics(K):
    # K is B x 3 x 3 or B x 4 x 4
    fx = K[:,0,0]
    fy = K[:,1,1]
    x0 = K[:,0,2]
    y0 = K[:,1,2]
    return fx, fy, x0, y0

def merge_intrinsics(fx, fy, x0, y0):
    B = list(fx.shape)[0]
    K = torch.zeros(B, 4, 4, dtype=torch.float32, device=fx.device)
    K[:,0,0] = fx
    K[:,1,1] = fy
    K[:,0,2] = x0
    K[:,1,2] = y0
    K[:,2,2] = 1.0
    K[:,3,3] = 1.0
    return K

def merge_rtlist(rlist, tlist):
    B, N, D, E = list(rlist.shape)
    assert(D==3)
    assert(E==3)
    B, N, F = list(tlist.shape)
    assert(F==3)

    # __p = lambda x: utils.basic.pack_seqdim(x, B)
    # __u = lambda x: utils.basic.unpack_seqdim(x, B)

    __p = lambda x: basic.pack_seqdim(x, B)
    __u = lambda x: basic.unpack_seqdim(x, B)


    rlist_, tlist_ = __p(rlist), __p(tlist)
    rtlist_ = merge_rt(rlist_, tlist_)
    rtlist = __u(rtlist_)
    return rtlist

def split_lrtlist(lrtlist):
    # splits a B x N x 19 tensor
    # into B x N x 3 (lens)
    # and B x N x 4 x 4 (rts)
    B, N, D = list(lrtlist.shape)
    assert(D==19)
    lenlist = lrtlist[:,:,:3].reshape(B, N, 3)
    ref_T_objs_list = lrtlist[:,:,3:].reshape(B, N, 4, 4)
    return lenlist, ref_T_objs_list

def merge_lrtlist(lenlist, rtlist):
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4
    # merges these into a B x N x 19 tensor
    B, N, D = list(lenlist.shape)
    assert(D==3)
    B2, N2, E, F = list(rtlist.shape)
    assert(B==B2)
    assert(N==N2)
    assert(E==4 and F==4)
    rtlist = rtlist.reshape(B, N, 16)
    lrtlist = torch.cat([lenlist, rtlist], axis=2)
    return lrtlist

def apply_4x4_to_lrtlist(Y_T_X, lrtlist_X):
    B, N, D = list(lrtlist_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)

    lenlist, rtlist_X = split_lrtlist(lrtlist_X)

    Y_T_Xs = Y_T_X.unsqueeze(1).repeat(1, N, 1, 1)
    Y_T_Xs_ = Y_T_Xs.view(B*N, 4, 4)
    rtlist_X_ = rtlist_X.reshape(B*N, 4, 4)
    rtlist_Y_ = basic.matmul2(Y_T_Xs_, rtlist_X_)

    rtlist_Y = rtlist_Y_.reshape(B, N, 4, 4)
    lrtlist_Y = merge_lrtlist(lenlist, rtlist_Y)
    return lrtlist_Y

def apply_4x4_to_lrt(Y_T_X, lrt_X):
    B, D = list(lrt_X.shape)
    assert(D==19)
    B2, E, F = list(Y_T_X.shape)
    assert(B2==B)
    assert(E==4 and F==4)

    return apply_4x4_to_lrtlist(Y_T_X, lrt_X.unsqueeze(1)).squeeze(1)

def get_xyzlist_from_lenlist(lenlist):
    B, N, D = list(lenlist.shape)
    assert(D==3)
    lx, ly, lz = torch.unbind(lenlist, axis=2)

    xs = torch.stack([lx/2., lx/2., -lx/2., -lx/2., lx/2., lx/2., -lx/2., -lx/2.], axis=2)
    ys = torch.stack([ly/2., ly/2., ly/2., ly/2., -ly/2., -ly/2., -ly/2., -ly/2.], axis=2)
    zs = torch.stack([lz/2., -lz/2., -lz/2., lz/2., lz/2., -lz/2., -lz/2., lz/2.], axis=2)

    # these are B x N x 8
    xyzlist = torch.stack([xs, ys, zs], axis=3)
    # this is B x N x 8 x 3
    return xyzlist

def get_xyzlist_from_lrtlist(lrtlist, include_clist=False):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = get_xyzlist_from_lenlist(lenlist)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 8, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 8, 3)

    if include_clist:
        clist_cam = get_clist_from_lrtlist(lrtlist).unsqueeze(2)
        xyzlist_cam = torch.cat([xyzlist_cam, clist_cam], dim=2)
    return xyzlist_cam

def get_clist_from_lrtlist(lrtlist):
    B, N, D = list(lrtlist.shape)
    assert(D==19)

    lenlist, rtlist = split_lrtlist(lrtlist)
    # lenlist is B x N x 3
    # rtlist is B x N x 4 x 4

    xyzlist_obj = torch.zeros((B, N, 1, 3), device=lrtlist.device)
    # xyzlist_obj is B x N x 8 x 3

    rtlist_ = rtlist.reshape(B*N, 4, 4)
    xyzlist_obj_ = xyzlist_obj.reshape(B*N, 1, 3)
    xyzlist_cam_ = apply_4x4(rtlist_, xyzlist_obj_)
    xyzlist_cam = xyzlist_cam_.reshape(B, N, 3)
    return xyzlist_cam

def wrap2pi(rad_angle):
    # puts the angle into the range [-pi, pi]
    return torch.atan2(torch.sin(rad_angle), torch.cos(rad_angle))


# https://github.com/lukasHoel/stylemesh/tree/cc5491e957771ddc16b92669bd406f8f7cf76746
def unproject(cam2world, intrinsic, depth):
    # get dimensions
    bs, _, H, W = depth.shape

    # create meshgrid with image dimensions (== pixel coordinates of source image)
    y = torch.linspace(0, H - 1, H).type_as(depth).int()
    x = torch.linspace(0, W - 1, W).type_as(depth).int()
    xx, yy = torch.meshgrid(x, y)
    xx = torch.transpose(xx, 0, 1).repeat(bs, 1, 1)
    yy = torch.transpose(yy, 0, 1).repeat(bs, 1, 1)

    # get intrinsics and depth in correct format to match image dimensions
    fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(1).expand_as(xx)
    cx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(1).expand_as(xx)
    fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(1).expand_as(yy)
    cy = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(1).expand_as(yy)
    depth = depth.squeeze()

    # inverse projection (K_inv) on pixel coordinates --> 3D point-cloud
    x = (xx - cx) / fx * depth
    y = (yy - cy) / fy * depth

    # combine each point into an (x,y,z,1) vector
    coords = torch.zeros(bs, H, W, 4).type_as(depth).float()
    coords[:, :, :, 0] = x
    coords[:, :, :, 1] = y
    coords[:, :, :, 2] = depth
    coords[:, :, :, 3] = 1

    # extrinsic view projection to target view
    coords = coords.view(bs, -1, 4)
    coords = torch.bmm(coords, cam2world)
    coords = coords.view(bs, H, W, 4)

    return coords


def reproject(cam2world_src, cam2world_tar, W, H, intrinsic, depth_src, depth_tar, color_tar, mask_tar):
    # get batch_size
    bs = mask_tar.shape[0]

    # calculate src2tar extrinsic matrix
    world2cam_tar = torch.inverse(cam2world_tar)
    src2tar = torch.transpose(torch.bmm(world2cam_tar, cam2world_src), 1, 2)

    # create meshgrid with image dimensions (== pixel coordinates of source image)
    y = torch.linspace(0, H - 1, H).type_as(color_tar).int()
    x = torch.linspace(0, W - 1, W).type_as(color_tar).int()
    xx, yy = torch.meshgrid(x, y)
    xx = torch.transpose(xx, 0, 1).repeat(bs, 1, 1)
    yy = torch.transpose(yy, 0, 1).repeat(bs, 1, 1)

    # get intrinsics and depth in correct format to match image dimensions
    fx = intrinsic[:,0,0].unsqueeze(1).unsqueeze(1).expand_as(xx)
    cx = intrinsic[:,0,2].unsqueeze(1).unsqueeze(1).expand_as(xx)
    fy = intrinsic[:,1,1].unsqueeze(1).unsqueeze(1).expand_as(yy)
    cy = intrinsic[:,1,2].unsqueeze(1).unsqueeze(1).expand_as(yy)
    depth_src = depth_src.squeeze()

    # inverse projection (K_inv) on pixel coordinates --> 3D point-cloud
    x = (xx - cx) / fx * depth_src
    y = (yy - cy) / fy * depth_src

    # combine each point into an (x,y,z,1) vector
    coords = torch.zeros(bs, H, W, 4).type_as(color_tar).float()
    coords[:, :, :, 0] = x
    coords[:, :, :, 1] = y
    coords[:, :, :, 2] = depth_src
    coords[:, :, :, 3] = 1

    # extrinsic view projection to target view
    coords = coords.view(bs, -1, 4)
    coords = torch.bmm(coords, src2tar)
    coords = coords.view(bs, H, W, 4)

    # projection (K) on 3D point-cloud --> pixel coordinates
    z_tar = coords[:, :, :, 2]
    x = coords[:, :, :, 0] / (1e-8 + z_tar) * fx + cx
    y = coords[:, :, :, 1] / (1e-8 + z_tar) * fy + cy

    # mask invalid pixel coordinates because of invalid source depth
    mask0 = (depth_src == 0)

    # mask invalid pixel coordinates after projection:
    # these coordinates are not visible in target view (out of screen bounds)
    mask1 = (x < 0) + (y < 0) + (x >= W - 1) + (y >= H - 1)  # 也要注意相机内参
    #
    # return (x, y, z_tar), mask1
    #




    # create 4 target pixel coordinates which map to the nearest integer coordinate
    # (left, top, right, bottom)
    lx = torch.floor(x).float()
    ly = torch.floor(y).float()
    rx = (lx + 1).float()
    ry = (ly + 1).float()

    def make_grid(x, y):
        """
        converts pixel coordinates from [0..W] or [0..H] to [-1..1] and stacks them together.
        :param x: x pixel coordinates with shape NxHxW
        :param y: y pixel coordinates with shape NxHxW
        :return: (x,y) pixel coordinate grid with shape NxHxWx2
        """
        x = (2.0 * x / W) - 1.0
        y = (2.0 * y / H) - 1.0
        grid = torch.stack((x, y), dim=3)
        return grid

    # combine to (x,y) pixel coordinates: (top-left, ..., bottom-right)
    ll = make_grid(lx, ly)
    lr = make_grid(lx, ry)
    rl = make_grid(rx, ly)
    rr = make_grid(rx, ry)

    # calculate difference between depth in target view after reprojection and gt depth in target view
    z_tar = z_tar.unsqueeze(1)
    sample_z1 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, ll,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z2 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, lr,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z3 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, rl,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z4 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, rr,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))

    # mask invalid pixel coordinates because of too high difference in depth
    mask2 = torch.minimum(torch.minimum(sample_z1, sample_z2), torch.minimum(sample_z3, sample_z4)) > 0.1
    mask2 = mask2.int().squeeze()

    # combine all masks
    mask_remap = (1 - (mask0 + mask1 + mask2 > 0).int()).float().unsqueeze(1)

    # create (x,y) pixel coordinate grid with reprojected float coordinates
    map_x = x.float()
    map_y = y.float()
    map = make_grid(map_x, map_y)

    # warp target rgb/mask to the new pixel coordinates based on the reprojection
    # also mask the results
    color_tar_to_src = torch.nn.functional.grid_sample(color_tar, map,
                                                                  mode="bilinear",
                                                                  padding_mode='border',
                                                                  align_corners=True)
    mask_tar = mask_tar.float().unsqueeze(1)
    mask = torch.nn.functional.grid_sample(mask_tar, map,
                                            mode="bilinear",
                                            padding_mode='border',
                                            align_corners=True)
    mask = (mask > 0.99) * mask_remap
    mask = mask.bool()
    color_tar_to_src *= mask

    return color_tar_to_src, mask.squeeze(1)


def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color


def mat2pose_vec(matrix: torch.Tensor):
    """
    Converts a 4x4 pose matrix into a 6-dof pose vector
    Args:
        matrix (ndarray): 4x4 pose matrix
    Returns:
        vector (ndarray): 6-dof pose vector comprising translation components (tx, ty, tz) and
        rotation components (rx, ry, rz)
    """

    # M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    rotx = torch.atan2(-matrix[..., 1, 2], matrix[..., 2, 2])

    # M[0, 2] = +siny, M[1, 2] = -sinx*cosy, M[2, 2] = +cosx*cosy
    cosy = torch.sqrt(matrix[..., 1, 2] ** 2 + matrix[..., 2, 2] ** 2)
    roty = torch.atan2(matrix[..., 0, 2], cosy)

    # M[0, 0] = +cosy*cosz, M[0, 1] = -cosy*sinz
    rotz = torch.atan2(-matrix[..., 0, 1], matrix[..., 0, 0])

    rotation = torch.stack((rotx, roty, rotz), dim=-1)

    # Extract translation params
    translation = matrix[..., :3, 3]
    return torch.cat((translation, rotation), dim=-1)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    return dist


# python /home/wsgan/project/bev/OccNeRF/utils/geom.py

if __name__ == '__main__':

    # pdb.set_trace()
    import time

    A = torch.ones((1, 10000, 3)).to('cuda')

    B = torch.ones((1, 100000, 3)).to('cuda')

    for i in range (1000):

        before_op_time = time.time()

        dist = square_distance(A, B)

        duration = time.time() - before_op_time

        print('duration:', duration)

    pass
