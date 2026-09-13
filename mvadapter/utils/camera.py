"""MV-Adapter のカメラ計算のうち、多視点画像の生成に要る部分だけを写したもの。

元: https://github.com/huanngzh/MV-Adapter (Apache-2.0) mvadapter/utils/mesh_utils/camera.py
元のファイルはメッシュ描画（nvdiffrast / trimesh）も抱えていて import が重いので、
正射影カメラの c2w を作る関数だけを残している。
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor

LIST_TYPE = Union[list, np.ndarray, torch.Tensor]


def list_to_pt(
    x: LIST_TYPE, dtype: Optional[torch.dtype] = None, device: Optional[str] = None
) -> torch.Tensor:
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=dtype, device=device)
    return x.to(dtype=dtype)


def get_c2w(
    elevation_deg: LIST_TYPE,
    distance: LIST_TYPE,
    azimuth_deg: Optional[LIST_TYPE],
    num_views: Optional[int] = 1,
    device: Optional[str] = None,
) -> torch.FloatTensor:
    if azimuth_deg is None:
        assert (
            num_views is not None
        ), "num_views must be provided if azimuth_deg is None."
        azimuth_deg = torch.linspace(
            0, 360, num_views + 1, dtype=torch.float32, device=device
        )[:-1]
    else:
        num_views = len(azimuth_deg)
    azimuth_deg = list_to_pt(azimuth_deg, dtype=torch.float32, device=device)
    elevation_deg = list_to_pt(elevation_deg, dtype=torch.float32, device=device)
    camera_distances = list_to_pt(distance, dtype=torch.float32, device=device)
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )
    center = torch.zeros_like(camera_positions)
    up = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)[None, :].repeat(
        num_views, 1
    )
    lookat = F.normalize(center - camera_positions, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w


def get_orthogonal_projection_matrix(
    batch_size: int,
    left: float,
    right: float,
    bottom: float,
    top: float,
    near: float = 0.1,
    far: float = 100.0,
    device: Optional[str] = None,
) -> torch.FloatTensor:
    projection_matrix = torch.zeros(
        batch_size, 4, 4, dtype=torch.float32, device=device
    )
    projection_matrix[:, 0, 0] = 2 / (right - left)
    projection_matrix[:, 1, 1] = -2 / (top - bottom)
    projection_matrix[:, 2, 2] = -2 / (far - near)
    projection_matrix[:, 0, 3] = -(right + left) / (right - left)
    projection_matrix[:, 1, 3] = -(top + bottom) / (top - bottom)
    projection_matrix[:, 2, 3] = -(far + near) / (far - near)
    projection_matrix[:, 3, 3] = 1
    return projection_matrix


@dataclass
class Camera:
    c2w: Optional[torch.FloatTensor]
    w2c: torch.FloatTensor
    proj_mtx: torch.FloatTensor
    mvp_mtx: torch.FloatTensor
    cam_pos: Optional[torch.FloatTensor]

    def __getitem__(self, index):
        if isinstance(index, int):
            sl = slice(index, index + 1)
        elif isinstance(index, slice):
            sl = index
        elif isinstance(index, list):
            sl = index
        else:
            raise NotImplementedError

        return Camera(
            c2w=self.c2w[sl] if self.c2w is not None else None,
            w2c=self.w2c[sl],
            proj_mtx=self.proj_mtx[sl],
            mvp_mtx=self.mvp_mtx[sl],
            cam_pos=self.cam_pos[sl] if self.cam_pos is not None else None,
        )

    def to(self, device: Optional[str] = None):
        if self.c2w is not None:
            self.c2w = self.c2w.to(device)
        self.w2c = self.w2c.to(device)
        self.proj_mtx = self.proj_mtx.to(device)
        self.mvp_mtx = self.mvp_mtx.to(device)
        if self.cam_pos is not None:
            self.cam_pos = self.cam_pos.to(device)

    def __len__(self):
        return self.c2w.shape[0]


def get_orthogonal_camera(
    elevation_deg: LIST_TYPE,
    distance: LIST_TYPE,
    left: float,
    right: float,
    bottom: float,
    top: float,
    azimuth_deg: Optional[LIST_TYPE] = None,
    num_views: Optional[int] = 1,
    near: float = 0.1,
    far: float = 100.0,
    device: Optional[str] = None,
):
    c2w = get_c2w(elevation_deg, distance, azimuth_deg, num_views, device)
    camera_positions = c2w[:, :3, 3]
    w2c = torch.linalg.inv(c2w)
    proj_mtx = get_orthogonal_projection_matrix(
        batch_size=c2w.shape[0],
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        near=near,
        far=far,
        device=device,
    )
    mvp_mtx = proj_mtx @ w2c
    return Camera(
        c2w=c2w, w2c=w2c, proj_mtx=proj_mtx, mvp_mtx=mvp_mtx, cam_pos=camera_positions
    )
