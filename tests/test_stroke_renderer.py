"""Compare pixels and derivatives against FRIDA's original per-stroke renderer."""

import json
import random
from types import SimpleNamespace
import warnings

import pytest
import torch
import torchgeometry

from brush_stroke import BrushStroke, rigid_body_transform
import painting as painting_module
from painting import Painting
from param2stroke import StrokeParametersToImage
from paint_utils3 import sort_brush_strokes_by_color, randomize_brush_stroke_order
from stroke_renderer import _warp_constants, warp_strokes

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def few_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def options(tmp_path):
    torch.manual_seed(17)
    torch.save(StrokeParametersToImage().state_dict(), tmp_path / 'param2img.pt')
    (tmp_path / 'param2stroke_settings.json').write_text(json.dumps(dict(
        w_p2i_m=.08, h_p2i_m=.042, xtra_room_horz_m=.01,
        xtra_room_vert_m=.001, MAX_BEND=.02)))
    return SimpleNamespace(cache_dir=str(tmp_path), CANVAS_WIDTH_M=.279,
        CANVAS_HEIGHT_M=.216, MAX_STROKE_LENGTH=.06, MIN_STROKE_LENGTH=.005,
        MIN_STROKE_Z=.1, MAX_ALPHA=.2, MAX_BEND=.02, render_batch_size=3)


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('size', [(31, 43), (64, 64), (256, 330)])
def test_warp_matches_legacy_pixels_and_gradients(device, size):
    torch.manual_seed(31)
    h, w = size
    inputs = [torch.rand(5, 1, h, w, device=device),
              torch.tensor([.17, -.36, 1.12, -2.03, .73], device=device),
              torch.tensor([-.94, .15, -.21, .96, -.1], device=device),
              torch.tensor([.91, -.13, .24, -.89, .37], device=device)]
    values = [x.clone().requires_grad_() for x in inputs]
    expected = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(5):
            matrix = rigid_body_transform(values[1][i], values[2][i] * w/2,
                                          values[3][i] * h/2, w/2, h/2)
            expected.append(torchgeometry.warp_perspective(values[0][i:i+1], matrix, (h, w)))
    expected = torch.cat(expected)
    weight = torch.rand_like(expected)
    expected_grads = torch.autograd.grad((expected * weight).mean(), values)
    actual_values = [x.clone().requires_grad_() for x in inputs]
    actual = warp_strokes(*actual_values)
    actual_grads = torch.autograd.grad((actual * weight).mean(), actual_values)
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-4)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, atol=3e-5, rtol=1e-3)


def make_painting(opt, device, ink=False):
    torch.manual_seed(9)
    strokes = [BrushStroke(opt, device=device, ink=ink,
               a=torch.tensor([.21 + i*.31]), xt=torch.tensor([-.12+i*.07]),
               yt=torch.tensor([.17-i*.08])) for i in range(5)]
    return Painting(opt, background_img=torch.rand(1, 3, 48, 62, device=device),
                    brush_strokes=strokes).to(device)


def rendered_and_grads(painting, batch, use_alpha, opacity):
    painting.render_batch_size = batch
    image, coverage = painting(48, 62, use_alpha=use_alpha,
                              return_alphas=True, opacity_factor=opacity)
    # Spatial weights expose gradients of positions/rotations and occlusion.
    weight = torch.linspace(.2, 1.1, image.numel(), device=image.device).reshape_as(image)
    loss = (image * weight).mean() + .07 * coverage.square().mean()
    grads = torch.autograd.grad(loss, tuple(painting.parameters()))
    return image, coverage, grads


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('batch', [1, 3, 64])
@pytest.mark.parametrize('use_alpha,opacity,ink', [(False, 1., False), (True, .6, False), (False, 1., True)])
def test_painting_matches_legacy_after_reordering(options, monkeypatch, device, batch, use_alpha, opacity, ink):
    monkeypatch.setattr(painting_module, 'device', torch.device(device))
    painting = make_painting(options, device, ink)
    sizes = []
    original_model = painting.param2img
    def tracked_model(parameters, *args):
        sizes.append(len(parameters))
        return original_model(parameters, *args)
    painting.param2img = tracked_model
    for ordering in ('initial', 'sorted', 'shuffled'):
        if ordering == 'sorted': sort_brush_strokes_by_color(painting, bin_size=3)
        if ordering == 'shuffled':
            random.seed(8)
            randomize_brush_stroke_order(painting)
        identities = list(map(id, painting.brush_strokes))
        sizes.clear()
        expected = rendered_and_grads(painting, 0, use_alpha, opacity)
        assert sizes == [1] * 5
        sizes.clear()
        actual = rendered_and_grads(painting, batch, use_alpha, opacity)
        assert sizes == [min(batch, 5-i) for i in range(0, 5, batch)]
        assert identities == list(map(id, painting.brush_strokes))
        for a, e in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(a, e, atol=6e-5, rtol=5e-4)
        for a, e in zip(actual[2], expected[2]):
            torch.testing.assert_close(a, e, atol=5e-5, rtol=1e-3)


@pytest.mark.parametrize('device', DEVICES)
def test_cache_is_constant_and_respects_size_device_dtype(device):
    _warp_constants.cache_clear()
    key = (48, 62, torch.device(device), torch.float32)
    first = _warp_constants(*key)
    assert all(a is b for a, b in zip(first, _warp_constants(*key)))
    assert all(not x.requires_grad and x.grad_fn is None and x.device.type == device for x in first)
    other_size = _warp_constants(49, 62, torch.device(device), torch.float32)
    other_dtype = _warp_constants(48, 62, torch.device(device), torch.float64)
    assert other_size[0].shape != first[0].shape
    assert other_dtype[0].dtype == torch.float64
    assert _warp_constants.cache_info().misses == 3


@pytest.mark.parametrize('settings', [{'render_batch_size': -1}, {'render_batch_size': True},
                                     {'cpu_threads': 0}, {'cpu_threads': True}, {'cpu_threads': 1.5}])
def test_remote_rejects_invalid_render_controls(settings):
    from remote.settings import plan_settings
    with pytest.raises(ValueError):
        plan_settings(settings)
