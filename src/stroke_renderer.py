"""Batched stroke images with the original torchgeometry coordinate convention.

Only stroke-independent tensors are cached. Sampling coordinates retain their
autograd history, and Painting still composites the strokes in their given order.
"""

from functools import lru_cache
import warnings

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, Resize
from torchgeometry.core.imgwarp import normal_transform_pixel
from torchgeometry.utils import create_meshgrid

from param2stroke import special_sigmoid


@lru_cache(maxsize=8)
def _warp_constants(h, w, device, dtype):
    """Match the legacy CPU-created coordinates exactly, transferring just once."""
    if min(h, w) < 2:
        raise ValueError('Stroke rendering requires at least two pixels per dimension')
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        grid = create_meshgrid(h, w).to(device=device, dtype=dtype)
        homogeneous = F.pad(grid.reshape(1, h * w, 2), (0, 1), value=1.)
        normalize = normal_transform_pixel(h, w).to(device=device, dtype=dtype)
        denormalize = torch.linalg.inv(normalize)
    return homogeneous, normalize, denormalize


def warp_strokes(images, angles, xt, yt):
    """Batched equivalent of RigidBodyTransformation.forward, including its pivot."""
    n, _, h, w = images.shape
    grid, normalize, denormalize = _warp_constants(h, w, images.device, images.dtype)
    a = -angles
    c, s = torch.cos(a), torch.sin(a)
    anchor_x, anchor_y = w / 2, h / 2
    tx = anchor_x - anchor_x * c + anchor_y * s + xt * (w / 2)
    ty = anchor_y - anchor_x * s - anchor_y * c + yt * (h / 2)
    zero, one = torch.zeros_like(a), torch.ones_like(a)
    matrix = torch.stack((c, -s, tx, s, c, ty, zero, zero, one), dim=-1).reshape(n, 3, 3)
    normalized = normalize @ (matrix @ denormalize)
    # Rigid transforms are nonsingular. inv_ex uses the same inverse operation
    # without a CPU error-check synchronization for every stroke.
    inverse = torch.linalg.inv_ex(normalized, check_errors=False).inverse
    points = torch.bmm(grid.expand(n, -1, -1), inverse.transpose(1, 2))
    sampling_grid = (points[..., :2] / points[..., 2:]).reshape(n, h, w, 2)
    # torchgeometry 0.1.2 uses a [-1, 1] grid with grid_sample's modern default
    # align_corners=False. Preserve this combination, including border behavior.
    return F.grid_sample(images, sampling_grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


def render_stroke_batches(strokes, param2img, h, w, batch_size):
    """Yield RGBA strokes in input order; the frozen renderer runs in eval mode."""
    for start in range(0, len(strokes), batch_size):
        batch = strokes[start:start + batch_size]
        parameters = torch.stack([torch.cat((s.stroke_length, s.stroke_bend,
                                             s.stroke_z, s.stroke_alpha)) for s in batch])
        masks = param2img(parameters, h, w).unsqueeze(1)
        if masks.shape[-2:] != (h, w):
            masks = Resize((h, w), InterpolationMode.BICUBIC, antialias=True)(masks)
        masks = warp_strokes(masks,
                             torch.cat([s.transformation.a for s in batch]),
                             torch.cat([s.transformation.xt for s in batch]),
                             torch.cat([s.transformation.yt for s in batch]))
        alpha = special_sigmoid(masks)
        colors = torch.stack([s.color_transform for s in batch])[:, :, None, None]
        rgba = torch.cat((alpha.expand(-1, 3, -1, -1) * 0 + colors, alpha), dim=1)
        yield from rgba.split(1, dim=0)
