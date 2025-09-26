import os
import torch
import numpy as np

def xyz_euler_to_four_by_four_matrix_batch(xyz_euler, relative=False, flatten=False, debug=False, euler_as_rotation=False):
    """
    Convert a batch of xyz euler angles to 4x4 transformation matrices.

    Args:
        xyz_euler (torch.Tensor): [B, 6], where each row is
            [x, y, z, rotx, roty, rotz].
        relative (bool): If True, make all frames relative to the first frame.
        flatten (bool): If True, return [B, 16] instead of [B, 4, 4].
        debug (bool): If True, return first frame repeat B times.

    Returns:
        torch.Tensor:
            [B, 4, 4] if flatten=False, else [B, 16].
    """
    batch_size = xyz_euler.size(0)

    # Split out components
    x, y, z, rotx, roty, rotz = torch.split(xyz_euler, 1, dim=1)
    rotx, roty, rotz = rotx * torch.pi / 180, roty * torch.pi / 180, rotz * torch.pi / 180

    zero = torch.zeros_like(x)
    one = torch.ones_like(x)

    # Build rotation matrices for each Euler angle
    Rx = torch.cat([
        one, zero, zero,
        zero, torch.cos(rotx), -torch.sin(rotx),
        zero, torch.sin(rotx), torch.cos(rotx)
    ], dim=1).view(batch_size, 3, 3)

    Ry = torch.cat([
        torch.cos(roty), zero, torch.sin(roty),
        zero, one, zero,
        -torch.sin(roty), zero, torch.cos(roty)
    ], dim=1).view(batch_size, 3, 3)

    Rz = torch.cat([
        torch.cos(rotz), -torch.sin(rotz), zero,
        torch.sin(rotz), torch.cos(rotz), zero,
        zero, zero, one
    ], dim=1).view(batch_size, 3, 3)

    # Combined rotation R = Rz * Ry * Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))

    # Translation
    T = torch.cat([x, y, z], dim=1).view(batch_size, 3, 1)

    # Assemble into 4x4 transformation matrix
    F = torch.cat([R, T], dim=2)  # [B, 3, 4]
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=F.dtype, device=F.device).view(1, 1, 4).expand(batch_size, -1, -1)
    F = torch.cat([F, bottom_row], dim=1)  # [B, 4, 4]

    if relative:
        R0 = F[0, :3, :3].unsqueeze(0)  # [1, 3, 3]
        t0 = F[0, :3, 3:].unsqueeze(0)  # [1, 3, 1]
        R0_inv = R0.transpose(1, 2)  # [1, 3, 3]

        R_all = F[:, :3, :3]  # [B, 3, 3]
        t_all = F[:, :3, 3:]  # [B, 3, 1]

        R0_inv_expanded = R0_inv.expand(batch_size, -1, -1)  # [B, 3, 3]
        R_rel = torch.bmm(R0_inv_expanded, R_all)  # [B, 3, 3]
        t_rel = torch.bmm(R0_inv_expanded, (t_all - t0))  # [B, 3, 1]

        F_rel = torch.cat([R_rel, t_rel], dim=2)  # [B, 3, 4]
        F_rel = torch.cat([F_rel, bottom_row], dim=1)  # [B, 4, 4]
        F = F_rel

    if debug:
        F = F[0].repeat(batch_size, 1, 1)

    if flatten:
        F = F.view(batch_size, 16)
    
    if euler_as_rotation:
        if relative:
            rotx = rotx - rotx[0]
            roty = roty - roty[0]
            rotz = rotz - rotz[0]
        translation_part = F[:, :3, 3]
        F = torch.cat([translation_part, rotx, roty, rotz], dim=1)

    return F