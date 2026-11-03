"""Removed objectives fail before loading inputs; retained image loss keeps gradients."""

import argparse
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest
import torch

from options import Options
from painting_optimization import load_objectives_data, parse_objective


@pytest.mark.parametrize('objective', [
    'style', 'dino', 'audio', 'speech', 'emotion', 'face', 'stable_diffusion', 'unknown',
])
def test_unsupported_objectives_rejected_before_loading_data(objective):
    parser = Options().initialize(argparse.ArgumentParser())
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(['--objective', objective])
    assert exc.value.code == 2

    opt = SimpleNamespace(objective=[objective], objective_data=['nonexistent-input'])
    with pytest.raises(ValueError, match='Unsupported painting objective'):
        load_objectives_data(opt)
    with pytest.raises(ValueError, match='Unsupported painting objective'):
        parse_objective(objective, None, None)


def test_l2_image_loading_preserves_weighted_gradients(tmp_path):
    path = tmp_path / 'target.png'
    Image.new('RGB', (8, 8), (128, 128, 128)).save(path)
    opt = SimpleNamespace(objective=['l2'], objective_data=[str(path)],
                          h_render=8, w_render=8, writer=Mock())
    load_objectives_data(opt)
    target, = opt.objective_data_loaded
    assert target.shape == (1, 3, 8, 8)
    torch.testing.assert_close(target, torch.full_like(target, 128 / 255))
    painting = torch.zeros_like(target, requires_grad=True)
    loss = parse_objective('l2', target, painting, weight=2)
    loss.backward()
    torch.testing.assert_close(loss.detach(), 2 * target.square().mean())
    torch.testing.assert_close(painting.grad, -4 * target / target.numel())
    opt.writer.add_image.assert_called_once()
