"""Silhouette plans must contain real, bounded strokes, not masked previews."""

from types import SimpleNamespace

import pytest
import torch

import painting as painting_module
from brush_stroke import BrushStroke
from foreground_planning import ForegroundPlanner
from painting import Painting


@pytest.fixture
def scene(monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(4)
    monkeypatch.setattr(painting_module, 'device', torch.device('cpu'))
    opt = SimpleNamespace(ink=False, objective=['l2'], h_render=64, w_render=64,
        CANVAS_WIDTH_M=.216, CANVAS_HEIGHT_M=.216, MIN_STROKE_LENGTH=.005,
        MAX_STROKE_LENGTH=.06, MIN_STROKE_Z=.1, MAX_BEND=.02, MAX_ALPHA=0.,
        render_batch_size=64, foreground_background_weight=4.,
        foreground_reconstruction_weight=1.)
    target = torch.ones(1, 3, 64, 64)
    target[:, :, 12:52, 12:52] = 0
    opt.objective_data_loaded = [target]

    def model(parameters, h, w):
        # A compact differentiable synthetic brush using the real affine renderer.
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        x = (xx - w/2) * opt.CANVAS_WIDTH_M / w
        y = (yy - h/2) * opt.CANVAS_HEIGHT_M / h
        length = parameters[:, 0, None, None]
        return torch.sigmoid((length/2 - (x - length/2).abs()) * 2000) * torch.sigmoid((.004 - y.abs()) * 2000)

    monkeypatch.setattr(painting_module, 'get_param2img', lambda *args, **kwargs: model)
    strokes = [BrushStroke(opt, color=torch.zeros(3)) for _ in range(9)]
    painting = Painting(opt, background_img=torch.ones_like(target), brush_strokes=strokes)
    planner = ForegroundPlanner(opt, painting, torch.zeros(1, 3))
    return opt, painting, planner


def test_foreground_initialization_and_reloadable_geometry(scene):
    opt, painting, planner = scene
    original_parameters = {id(s) for s in painting.brush_strokes}
    planner.initialize(painting)
    assert 0 < len(painting) <= 9
    assert all(id(s) in original_parameters for s in painting.brush_strokes)
    with torch.no_grad():
        alphas = planner._alphas(painting, list(painting.brush_strokes))
        assert planner._valid(alphas).all()
        # The preview uses precisely the unmasked brush alphas.
        expected = (1 - alphas).prod(0)
        torch.testing.assert_close(painting(64, 64, use_alpha=False)[0], expected.expand(3, -1, -1))
    assert all(torch.equal(s.color_transform, torch.zeros(3)) for s in painting.brush_strokes)
    assert all(.006999 <= s.stroke_length.item() <= .058001 for s in painting.brush_strokes)


def test_final_pruning_removes_background_stroke_preserves_order(scene):
    _, painting, planner = scene
    planner.initialize(painting)
    with torch.no_grad():
        bad = painting.brush_strokes[0]
        bad.transformation.xt.fill_(-.85)
        bad.transformation.yt.fill_(-.85)
    before = list(painting.brush_strokes)
    planner.refine(painting, step=400, relocate=False)
    assert not any(s is bad for s in painting.brush_strokes)
    assert list(painting.brush_strokes) == [s for s in before if any(s is k for k in painting.brush_strokes)]
    assert planner._valid(planner._alphas(painting, list(painting.brush_strokes))).all()


def test_background_loss_has_full_canvas_gradients(scene):
    opt, _, planner = scene
    opt.objective = ['clip_conv_loss']
    image = torch.full((1, 3, 64, 64), .5, requires_grad=True)
    planner.loss(image).backward()
    assert image.grad[0, 0, 0, 0] < 0  # Lighten white background, even at corners.
    assert image.grad[0, 0, 32, 32] > 0  # Darken the silhouette.
    assert image.grad[0, 0, 0, 0].abs() > image.grad[0, 0, 32, 32].abs()


def test_relocation_reuses_stroke_and_clears_old_optimizer_state(scene):
    _, painting, planner = scene
    planner.initialize(painting)
    stroke = painting.brush_strokes[0]
    with torch.no_grad():
        stroke.transformation.xt.fill_(-.85)
        stroke.transformation.yt.fill_(-.85)
    optimizers = painting.get_optimizers()
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for p in group['params']:
                optimizer.state[p]['old_position'] = True
    planner.refine(painting, optimizers, step=50)
    assert planner.stats['relocated_strokes'] > 0
    assert any(s is stroke for s in painting.brush_strokes)
    assert all(p not in optimizer.state for optimizer in optimizers for p in stroke.parameters())
    assert planner._valid(planner._alphas(painting, list(painting.brush_strokes))).all()


def test_invalid_target_palette_and_background_rejected(scene):
    opt, painting, _ = scene
    with pytest.raises(ValueError, match='black paint'):
        ForegroundPlanner(opt, painting, torch.ones(1, 3))
    opt.objective_data_loaded[0].fill_(1)
    with pytest.raises(ValueError, match='foreground and background'):
        ForegroundPlanner(opt, painting, torch.zeros(1, 3))


