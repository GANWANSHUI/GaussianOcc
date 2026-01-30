import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import geom
from utils import basic
from utils import render


def world2contracted(xyz_world, pc_range_roi=[-52, -52, 0, 52, 52, 6], ratio=0.8):
    """
    Convert 3D world coordinates to a contracted coordinate system based on a specified ROI.

    Args:
        xyz_world (torch.Tensor): Input tensor with shape [..., 3] representing 3D world coordinates.
        pc_range_roi (list, optional): List of 6 elements defining the ROI. Default is [-52, -52, 0, 52, 52, 6].

    Returns:
        torch.Tensor: Tensor with shape [..., 3] representing coordinates in the contracted system.
    """
    xyz_min = torch.tensor(pc_range_roi[:3]).to(xyz_world).reshape([1]*len(xyz_world.shape[:-1]) + [3])
    xyz_max = torch.tensor(pc_range_roi[3:]).to(xyz_world).reshape([1]*len(xyz_world.shape[:-1]) + [3])
    t = ratio / (1 - ratio)
    xyz_scaled = (2 * (xyz_world - xyz_min) / (xyz_max - xyz_min) - 1) * t
    xyz_abs = torch.abs(xyz_scaled)
    xyz_contracted = torch.where(
        xyz_abs <= t,
        xyz_scaled,
        xyz_scaled.sign() * (1.0 + t - 1.0/(xyz_abs + 1 - t))
    )
    return xyz_contracted / (t + 1) # range: [-1, 1]


def contracted2world(xyz_contracted, pc_range_roi=[-80, -80, -3, 80, 80, 8], ratio=0.8):
    """
    Convert 3D contracted coordinates back to the world coordinate system based on a specified ROI.

    Args:
        xyz_contracted (torch.Tensor): Input tensor with shape [..., 3] representing 3D contracted coordinates.
        pc_range_roi (list, optional): List of 6 elements defining the ROI. Default is [-52, -52, 0, 52, 52, 6].

    Returns:
        torch.Tensor: Tensor with shape [..., 3] representing coordinates in the world system.
    """
    xyz_min = torch.tensor(pc_range_roi[:3]).to(xyz_contracted).reshape([1]*len(xyz_contracted.shape[:-1]) + [3])
    xyz_max = torch.tensor(pc_range_roi[3:]).to(xyz_contracted).reshape([1]*len(xyz_contracted.shape[:-1]) + [3])
    t = ratio / (1 - ratio)
    xyz_ = xyz_contracted * (t + 1)
    xyz_abs = torch.abs(xyz_)
    xyz_scaled = torch.where(
        xyz_abs <= t,
        xyz_,
        xyz_.sign() * (t - 1.0 + 1.0/(t + 1 - xyz_abs))
    ) / t
    xyz_world = 0.5 * (xyz_scaled + 1) * (xyz_max - xyz_min) + xyz_min
    return xyz_world


class Vox_util(nn.Module):
    def __init__(self, Z, Y, X, scene_centroid, bounds, position = 'embedding', length_pose_encoding = 3, opt = None, pad=None, assert_cube=False):

        super(Vox_util, self).__init__()

        self.opt = opt

        self.XMIN, self.XMAX, self.YMIN, self.YMAX, self.ZMIN, self.ZMAX = bounds

        self.Z, self.Y, self.X = Z, Y, X # 16, 256, 256

        self.max_depth = math.sqrt(self.XMAX*self.XMAX + self.YMAX*self.YMAX + self.ZMAX*self.ZMAX)
        
        self.pc_range_roi = [self.opt.real_size[0], self.opt.real_size[2], self.opt.real_size[4], \
                             self.opt.real_size[1], self.opt.real_size[3], self.opt.real_size[5]]  # [x_min, y_min, z_min, x_max, y_max, z_max]


        scene_centroid = scene_centroid.detach().cpu().numpy()
        x_centroid, y_centroid, z_centroid = scene_centroid[0]
        self.XMIN += x_centroid
        self.XMAX += x_centroid
        self.YMIN += y_centroid
        self.YMAX += y_centroid
        self.ZMIN += z_centroid
        self.ZMAX += z_centroid

        self.default_vox_size_X = (self.XMAX-self.XMIN)/float(X)
        self.default_vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        self.default_vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)

        if pad:
            Z_pad, Y_pad, X_pad = pad
            self.ZMIN -= self.default_vox_size_Z * Z_pad
            self.ZMAX += self.default_vox_size_Z * Z_pad
            self.YMIN -= self.default_vox_size_Y * Y_pad
            self.YMAX += self.default_vox_size_Y * Y_pad
            self.XMIN -= self.default_vox_size_X * X_pad
            self.XMAX += self.default_vox_size_X * X_pad


        # for embedding
        self.length_pose_encoding = length_pose_encoding
        self.position = position
        self.register_buffer('posfreq', torch.FloatTensor([(2 ** i) for i in range(length_pose_encoding)]))

        if assert_cube:
            # we assume cube voxels
            if (not np.isclose(self.default_vox_size_X, self.default_vox_size_Y)) or (not np.isclose(self.default_vox_size_X, self.default_vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                )
                print('self.default_vox_size_X', self.default_vox_size_X)
                print('self.default_vox_size_Y', self.default_vox_size_Y)
                print('self.default_vox_size_Z', self.default_vox_size_Z)
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Y))
            assert(np.isclose(self.default_vox_size_X, self.default_vox_size_Z))

    def Ref2Mem(self, xyz, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in ref coordinates
        # transforms ref coordinates into mem coordinates
        B, N, C = list(xyz.shape)
        device = xyz.device
        assert(C==3)
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        xyz = geom.apply_4x4(mem_T_ref, xyz)
        return xyz

    def Mem2Ref(self, xyz_mem, Z, Y, X, assert_cube=False):
        # xyz is B x N x 3, in mem coordinates
        # transforms mem coordinates into ref coordinates
        B, N, C = list(xyz_mem.shape)
        
        if self.opt.contracted_coord:
            # Avoid inf
            torch.clamp_(xyz_mem[:, :, 0], 0.5, X - 1.5)
            torch.clamp_(xyz_mem[:, :, 1], 0.5, Y - 1.5)
            torch.clamp_(xyz_mem[:, :, 2], 0.05, Z - 1.05)
            
            xyz_norm = xyz_mem / torch.tensor([X - 1, Y - 1, Z - 1], dtype=torch.float32, device=xyz_mem.device).view(1, 1, 3) * 2 - 1  # value range: [-2, 2]
            xyz_ref = contracted2world(xyz_norm, pc_range_roi=self.pc_range_roi, ratio=self.opt.contracted_ratio)
            
        else:
            ref_T_mem = self.get_ref_T_mem(B, Z, Y, X, assert_cube=assert_cube, device=xyz_mem.device)
            xyz_ref = geom.apply_4x4(ref_T_mem, xyz_mem)

        return xyz_ref



    def get_mem_T_ref(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        vox_size_X = (self.XMAX-self.XMIN)/float(X)
        vox_size_Y = (self.YMAX-self.YMIN)/float(Y)
        vox_size_Z = (self.ZMAX-self.ZMIN)/float(Z)

        if assert_cube:
            if (not np.isclose(vox_size_X, vox_size_Y)) or (not np.isclose(vox_size_X, vox_size_Z)):
                print('Z, Y, X', Z, Y, X)
                print('bounds for this iter:',
                      'X = %.2f to %.2f' % (self.XMIN, self.XMAX),
                      'Y = %.2f to %.2f' % (self.YMIN, self.YMAX),
                      'Z = %.2f to %.2f' % (self.ZMIN, self.ZMAX),
                )
                print('vox_size_X', vox_size_X)
                print('vox_size_Y', vox_size_Y)
                print('vox_size_Z', vox_size_Z)
            assert(np.isclose(vox_size_X, vox_size_Y))
            assert(np.isclose(vox_size_X, vox_size_Z))

        # translation
        # (this makes the left edge of the leftmost voxel correspond to XMIN)
        center_T_ref = geom.eye_4x4(B, device=device)
        center_T_ref[:,0,3] = -self.XMIN-vox_size_X/2.0
        center_T_ref[:,1,3] = -self.YMIN-vox_size_Y/2.0
        center_T_ref[:,2,3] = -self.ZMIN-vox_size_Z/2.0


        # scaling
        # (this makes the right edge of the rightmost voxel correspond to XMAX)
        mem_T_center = geom.eye_4x4(B, device=device)
        mem_T_center[:,0,0] = 1./vox_size_X
        mem_T_center[:,1,1] = 1./vox_size_Y
        mem_T_center[:,2,2] = 1./vox_size_Z
        mem_T_ref = basic.matmul2(mem_T_center, center_T_ref)

        return mem_T_ref

    def get_ref_T_mem(self, B, Z, Y, X, assert_cube=False, device='cuda'):
        mem_T_ref = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=device)
        # note safe_inverse is inapplicable here,
        # since the transform is nonrigid
        ref_T_mem = mem_T_ref.inverse()
        return ref_T_mem

    def get_inbounds(self, xyz, Z, Y, X, already_mem=False, padding=0.0, assert_cube=False):
        # xyz is B x N x 3
        # padding should be 0 unless you are trying to account for some later cropping
        if not already_mem:
            xyz = self.Ref2Mem(xyz, Z, Y, X, assert_cube=assert_cube)

        x = xyz[:,:,0]
        y = xyz[:,:,1]
        z = xyz[:,:,2]

        x_valid = ((x-padding)>-0.5).byte() & ((x+padding)<float(X-0.5)).byte()
        y_valid = ((y-padding)>-0.5).byte() & ((y+padding)<float(Y-0.5)).byte()
        z_valid = ((z-padding)>-0.5).byte() & ((z+padding)<float(Z-0.5)).byte()
        nonzero = (~(z==0.0)).byte()

        inbounds = x_valid & y_valid & z_valid & nonzero
        return inbounds.bool()

    def voxelize_xyz(self, xyz_ref, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        assert(D==3)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:,0:1]*0, Z, Y, X, assert_cube=assert_cube)
        vox = self.get_occupancy(xyz_mem, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return vox

    def voxelize_xyz_and_feats(self, xyz_ref, feats, Z, Y, X, already_mem=False, assert_cube=False, clean_eps=0):
        B, N, D = list(xyz_ref.shape)
        B2, N2, D2 = list(feats.shape)
        assert(D==3)
        assert(B==B2)
        assert(N==N2)
        if already_mem:
            xyz_mem = xyz_ref
        else:
            xyz_mem = self.Ref2Mem(xyz_ref, Z, Y, X, assert_cube=assert_cube)
            xyz_zero = self.Ref2Mem(xyz_ref[:,0:1]*0, Z, Y, X, assert_cube=assert_cube)
        feats = self.get_feat_occupancy(xyz_mem, feats, Z, Y, X, clean_eps=clean_eps, xyz_zero=xyz_zero)
        return feats

    def get_occupancy(self, xyz, Z, Y, X, clean_eps=0, xyz_zero=None):
        # xyz is B x N x 3 and in mem coords
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        assert(C==3)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero
            dist = torch.norm(xyz_zero-xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz) # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x*mask
        y = y*mask
        z = z*mask

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        voxels = torch.zeros(B*Z*Y*X, device=xyz.device).float()
        voxels[vox_inds.long()] = 1.0
        # zero out the singularity
        voxels[base.long()] = 0.0
        voxels = voxels.reshape(B, 1, Z, Y, X)
        # B x 1 x Z x Y x X
        return voxels

    def get_feat_occupancy(self, xyz, feat, Z, Y, X, clean_eps=0, xyz_zero=None):
        # xyz is B x N x 3 and in mem coords
        # feat is B x N x D
        # we want to fill a voxel tensor with 1's at these inds
        B, N, C = list(xyz.shape)
        B2, N2, D2 = list(feat.shape)
        assert(C==3)
        assert(B==B2)
        assert(N==N2)

        # these papers say simple 1/0 occupancy is ok:
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3d_CVPR_2018_paper.pdf
        #  http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf
        # cont fusion says they do 8-neighbor interp
        # voxelnet does occupancy but with a bit of randomness in terms of the reflectance value i think

        inbounds = self.get_inbounds(xyz, Z, Y, X, already_mem=True)
        x, y, z = xyz[:,:,0], xyz[:,:,1], xyz[:,:,2]
        mask = torch.zeros_like(x)
        mask[inbounds] = 1.0

        if xyz_zero is not None:
            # only take points that are beyond a thresh of zero
            dist = torch.norm(xyz_zero-xyz, dim=2)
            mask[dist < 0.1] = 0

        if clean_eps > 0:
            # only take points that are already near centers
            xyz_round = torch.round(xyz) # B, N, 3
            dist = torch.norm(xyz_round - xyz, dim=2)
            mask[dist > clean_eps] = 0

        # set the invalid guys to zero
        # we then need to zero out 0,0,0
        # (this method seems a bit clumsy)
        x = x*mask # B, N
        y = y*mask
        z = z*mask
        feat = feat*mask.unsqueeze(-1) # B, N, D

        x = torch.round(x)
        y = torch.round(y)
        z = torch.round(z)
        x = torch.clamp(x, 0, X-1).int()
        y = torch.clamp(y, 0, Y-1).int()
        z = torch.clamp(z, 0, Z-1).int()

        # permute point orders
        perm = torch.randperm(N)
        x = x[:, perm]
        y = y[:, perm]
        z = z[:, perm]
        feat = feat[:, perm]

        x = x.view(B*N)
        y = y.view(B*N)
        z = z.view(B*N)
        feat = feat.view(B*N, -1)

        dim3 = X
        dim2 = X * Y
        dim1 = X * Y * Z

        base = torch.arange(0, B, dtype=torch.int32, device=xyz.device)*dim1
        base = torch.reshape(base, [B, 1]).repeat([1, N]).view(B*N)

        vox_inds = base + z * dim2 + y * dim3 + x
        feat_voxels = torch.zeros((B*Z*Y*X, D2), device=xyz.device).float()
        feat_voxels[vox_inds.long()] = feat
        # zero out the singularity
        feat_voxels[base.long()] = 0.0
        feat_voxels = feat_voxels.reshape(B, Z, Y, X, D2).permute(0, 4, 1, 2, 3)
        # B x C x Z x Y x X
        return feat_voxels

    def unproject_image_to_mem(self, rgb_camB, pixB_T_camA, camB_T_camA, Z, Y, X, assert_cube=False):
        """2D to 3D lifting

        Args:
            rgb_camB (torch.Tensor): Image features. [B, C, H // 4, W // 4] (6, 64, 84, 168)
            pixB_T_camA (torch.Tensor): Projection matrix. From world coordinates to pixel coordinates. [B, 4, 4] (6, 4, 4)
            camB_T_camA (torch.Tensor): Projection matrix. From world coordinates to camera coordinates. [B, 4, 4] (6, 4, 4)
            Z, Y, X (int): Voxel size.

        Returns:
            torch.Tensor: Voxel features. [B, C, Z, Y, X] (6, 64, 16, 256, 256)
        """
        B, C, H, W = list(rgb_camB.shape)

        xyz_memA = basic.gridcloud3d(B, Z, Y, X, norm=False, device=pixB_T_camA.device)
        # y is height here

        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)  # xyz in world coordinates

        xyz_camB = geom.apply_4x4(camB_T_camA, xyz_camA)  # xyz in camera coordinates

        z = xyz_camB[:,:,2]  # z is depth

        xyz_pixB = geom.apply_4x4(pixB_T_camA, xyz_camA)
        normalizer = torch.unsqueeze(xyz_pixB[:,:,2], 2)
        EPS=1e-6
        # z = xyz_pixB[:,:,2]
        xy_pixB = xyz_pixB[:,:,:2]/torch.clamp(normalizer, min=EPS)  # xyz in pixel coordinates
        # this is B x N x 2
        # this is the (floating point) pixel coordinate of each voxel
        x, y = xy_pixB[:,:,0], xy_pixB[:,:,1]
        # these are B x N

        x_valid = (x>-0.5).bool() & (x<float(W-0.5)).bool()
        y_valid = (y>-0.5).bool() & (y<float(H-0.5)).bool()
        z_valid = (z>0.0).bool()
        valid_mem = (x_valid & y_valid & z_valid).reshape(B, 1, Z, Y, X).float()


        # native pytorch version
        y_pixB, x_pixB = basic.normalize_grid2d(y, x, H, W)  # normalize to [-1, 1]
        # since we want a 3d output, we need 5d tensors
        z_pixB = torch.zeros_like(x)
        xyz_pixB = torch.stack([x_pixB, y_pixB, z_pixB], axis=2)

        rgb_camB = rgb_camB.unsqueeze(2)
        xyz_pixB = torch.reshape(xyz_pixB, [B, Z, Y, X, 3])
        values = F.grid_sample(rgb_camB, xyz_pixB, align_corners=False) # (N, H, W, Z, 3)

        values = torch.reshape(values, (B, C, Z, Y, X))
        # 16, 256, 256
        values = values * valid_mem

        return values
    

    def get_voxel_grid(self, cam_center, assert_cube=False):
        
        # Z, Y, X = self.Z, self.Y, self.X

        if self.opt.gs_sample == 0:
            Z, Y, X = self.Z, self.Y, self.X

        else:
            Z, Y, X = int(self.opt.gs_sample*self.Z), int(self.opt.gs_sample*self.Y), int(self.opt.gs_sample*self.X)

        xyz_memA = basic.gridcloud3d(self.opt.B, Z, Y, X, norm=False, device=cam_center.device)  # 定义voxel大小
        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)  # re-center

        return xyz_camA, Z, Y, X 


    def get_meta_data(self, cam_center, camB_T_camA = None, abs_position=False, assert_cube=False):

        Z, Y, X = self.Z, self.Y, self.X
        xyz_memA = basic.gridcloud3d(self.opt.B, Z, Y, X, norm=False, device=cam_center.device)  #
        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)  # re-center

        # meta depth
        meta_depth = (xyz_camA - cam_center.unsqueeze(1)).norm(dim=2).unsqueeze(-1) / (self.max_depth*1.5)

        # meta angle
        if camB_T_camA is not None:
            xyz_camB = geom.apply_4x4(camB_T_camA, xyz_camA)
            # get mask
            mask = torch.logical_and(torch.logical_and(xyz_camB[:, :, 0] > self.XMIN + 1 , xyz_camB[:, :, 0] < self.XMAX - 1),
                                    torch.logical_and(xyz_camB[:, :, 1] > self.YMIN + 1, xyz_camB[:, :, 1] < self.YMAX - 1))

            meta_mask = torch.logical_and(torch.logical_and(xyz_camB[:, :, 2] > self.ZMIN, xyz_camB[:, :, 2] < self.ZMAX), mask)
            meta_mask = meta_mask.unsqueeze(-1)

            cur_points_rays_bk3hw = xyz_camA - cam_center
            src_points_rays_bk3hw = xyz_camB - cam_center
            meta_angle = F.cosine_similarity(cur_points_rays_bk3hw, src_points_rays_bk3hw, dim=2, eps=1e-5).unsqueeze(-1)

            # meta_position [-1, 1]
            xyz_camA[..., 0] = xyz_camA[..., 0] / self.XMAX
            xyz_camA[..., 1] = xyz_camA[..., 1] / self.YMAX

            # [0, 1]
            # xyz_camA[..., 2] = 2.0 * (xyz_camA[..., 2] / self.ZMAX) - 1.0
            xyz_camA[..., 2] = xyz_camA[..., 2] / self.ZMAX

            if abs_position:
                meta_position = abs(xyz_camA)
            else:
                meta_position = xyz_camA

            # meta_position_emb = (meta_position.unsqueeze(-1) * self.posfreq).flatten(-2)
            # meta_position = torch.cat([meta_position, meta_position_emb.sin(), meta_position_emb.cos()], -1)
            # print('meta shape', meta_depth.shape, meta_mask.shape, meta_angle.shape, meta_position.shape)
            meta_data = torch.cat([meta_depth, meta_mask, meta_angle, meta_position], dim=-1)
            return meta_data

        else:
            # meta_position [-1, 1]
            xyz_camA[..., 0] = xyz_camA[..., 0] / self.XMAX
            xyz_camA[..., 1] = xyz_camA[..., 1] / self.YMAX

            # [0, 1]
            xyz_camA[..., 2] = xyz_camA[..., 2] / self.ZMAX

            if abs_position:
                meta_position = abs(xyz_camA)
            else:
                meta_position = xyz_camA

            meta_data = torch.cat([meta_depth, meta_position], dim=-1)

            return meta_data


    def get_voxel_position(self, cam_center, abs_position=True, assert_cube=False):
        Z, Y, X = self.Z, self.Y, self.X
        xyz_memA = basic.gridcloud3d(1, Z, Y, X, norm=False, device=cam_center.device)  # 定义voxel大小
        xyz_camA = self.Mem2Ref(xyz_memA, Z, Y, X, assert_cube=assert_cube)  # re-center
        return xyz_camA.squeeze(0)


    def apply_mem_T_ref_to_lrtlist(self, lrtlist_cam, Z, Y, X, assert_cube=False):
        # lrtlist is B x N x 19, in cam coordinates
        # transforms them into mem coordinates, including a scale change for the lengths
        B, N, C = list(lrtlist_cam.shape)
        assert(C==19)
        mem_T_cam = self.get_mem_T_ref(B, Z, Y, X, assert_cube=assert_cube, device=lrtlist_cam.device)


    
    def get_spring_meta_data(self, cam_center, xyz_camA = None, abs_position=False, assert_cube=False):
        # 这里省略了很多注释
        if abs_position:
            meta_position = abs(xyz_camA)
        else:
            meta_position = xyz_camA

        meta_data = meta_position

        return meta_data