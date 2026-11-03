"""Plan executable black strokes for a silhouette on a white canvas.

Masks score complete learned brush footprints; they never clip rendered pixels.
The requested stroke count is a budget. Proposals reuse the same physical stroke
parameters and renderer as ordinary FRIDA planning.
"""

import copy
import math

import torch
from torch import nn

from brush_stroke import BrushStroke
from stroke_renderer import render_stroke_batches


class ForegroundPlanner:
    # At most 2% of a retained stroke's predicted paint may fall outside the
    # target silhouette. This includes stroke width and learned brush texture.
    max_background_fraction = .02
    improvement_epsilon = 1e-7
    proposal_count = 8

    def __init__(self, opt, painting, palette):
        if (opt.ink or palette is None or palette.shape != (1, 3)
                or not torch.equal(palette, torch.zeros_like(palette))):
            raise ValueError('foreground_mode requires one fixed black paint color')
        if len(opt.objective) != 1 or opt.objective[0] not in ('l2', 'clip_conv_loss', 'clip_fc_loss'):
            raise ValueError('foreground_mode requires one image reconstruction objective')
        target = opt.objective_data_loaded[0].detach()
        if target.shape != (1, 3, opt.h_render, opt.w_render):
            raise ValueError('foreground_mode requires an RGB image target')
        if (target.max(1).values - target.min(1).values).mean() > .02:
            raise ValueError('foreground_mode requires a grayscale silhouette')
        for name in ('foreground_background_weight', 'foreground_reconstruction_weight'):
            value = getattr(opt, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not torch.all(painting.background_img[:, :3] >= .999):
            raise ValueError('foreground_mode currently requires a white starting canvas')
        self.target = target.mean(1)[0]
        self.foreground = self.target < .5
        if not self.foreground.any() or self.foreground.all():
            raise ValueError('foreground_mode requires both foreground and background pixels')
        # Reject photos rather than silently interpreting their gray areas as a mask.
        if ((self.target > .1) & (self.target < .9)).float().mean() > .1:
            raise ValueError('foreground_mode expects a black-and-white silhouette')
        self.h, self.w = self.target.shape
        self.opt = opt
        self.weights = 1 + opt.foreground_background_weight * (~self.foreground)
        self.stats = dict(requested_strokes=len(painting), initialized_strokes=0,
                          initialization_rejected=0, relocated_strokes=0,
                          removed_strokes=0, refinements=[],
                          max_background_fraction=self.max_background_fraction)

    def loss(self, rgb):
        error = (rgb - self.target[None, None]).square()
        result = self.opt.foreground_background_weight * (error * ~self.foreground).mean()
        if self.opt.objective[0] != 'l2':
            result = result + self.opt.foreground_reconstruction_weight * error.mean()
        return result

    def _cost(self, canvas):
        return ((canvas - self.target).square() * self.weights).mean(dim=(-2, -1))

    def _alphas(self, painting, strokes):
        return torch.cat([s[:, 3] for s in render_stroke_batches(
            strokes, painting.param2img, self.h, self.w, 64)], dim=0)

    def _valid(self, alphas):
        # Ignore the renderer's tiny sigmoid floor only for mass accounting.
        mass = (alphas - 1.2e-7).clamp_min(0)
        total = mass.sum(dim=(-2, -1))
        outside = (mass * ~self.foreground).sum(dim=(-2, -1))
        return (total > 1) & (outside <= total * self.max_background_fraction)

    def _copy_geometry(self, destination, source):
        for name in ('stroke_length', 'stroke_bend', 'stroke_z', 'stroke_alpha'):
            getattr(destination, name).copy_(getattr(source, name))
        for name in ('a', 'xt', 'yt'):
            getattr(destination.transformation, name).copy_(getattr(source.transformation, name))
        destination.color_transform.zero_()

    def _propose(self, painting, stroke, canvas):
        # Sample dark target pixels that are still light in the current painting.
        residual = (canvas - self.target).clamp_min(0).square() * self.foreground
        if residual.sum() < 1e-5:
            return None
        locations = torch.multinomial(residual.flatten(), self.proposal_count, replacement=True)
        random = torch.rand(self.proposal_count, 4, device=canvas.device)
        candidates = []
        for i in range(self.proposal_count):
            candidate = copy.deepcopy(stroke)
            candidate.transformation.xt.copy_((locations[i] % self.w).reshape(1) * (2 / self.w) - 1)
            candidate.transformation.yt.copy_((locations[i] // self.w).reshape(1) * (2 / self.h) - 1)
            candidate.transformation.a.copy_((random[i, 0] * (2 * torch.pi)).reshape(1))
            low = stroke.MIN_STROKE_LENGTH + .002
            high = stroke.MAX_STROKE_LENGTH - .002
            # Include short strokes for narrow legs and boundary regions.
            candidate.stroke_length.fill_(low)
            if i < self.proposal_count // 2:
                candidate.stroke_length.add_(random[i, 1] * (high - low))
            candidate.stroke_bend.zero_()
            candidate.stroke_z.copy_((stroke.MIN_STROKE_Z + random[i, 2] * .4).reshape(1))
            candidate.stroke_alpha.zero_()
            BrushStroke.make_valid(candidate)
            candidate.color_transform.zero_()
            candidates.append(candidate)
        alpha = self._alphas(painting, candidates)
        costs = self._cost(canvas * (1 - alpha))
        costs.masked_fill_(~self._valid(alpha), torch.inf)
        index = int(costs.argmin())
        if costs[index] >= self._cost(canvas) - self.improvement_epsilon:
            return None
        return candidates[index], alpha[index], costs[index]

    @torch.no_grad()
    def initialize(self, painting):
        canvas = torch.ones_like(self.target)
        kept = []
        for stroke in painting.brush_strokes:
            # BrushStroke's random constructor can give bend and length shared
            # storage when it clips bend to length. Independent proposal edits
            # must not zero the length when straightening the bend.
            for parameter in stroke.parameters():
                parameter.set_(parameter.detach().clone())
            proposal = self._propose(painting, stroke, canvas)
            if proposal is None:
                self.stats['initialization_rejected'] += 1
                continue
            candidate, alpha, _ = proposal
            self._copy_geometry(stroke, candidate)
            kept.append(stroke)
            canvas *= 1 - alpha
        if not kept:
            raise ValueError('No valid brush footprints fit this silhouette; check brush/canvas scale')
        painting.brush_strokes = nn.ModuleList(kept)
        self.stats['initialized_strokes'] = len(kept)

    @torch.no_grad()
    def refine(self, painting, optimizers=(), *, step, relocate=True):
        strokes = list(painting.brush_strokes)
        if not strokes:
            return
        alpha = self._alphas(painting, strokes)
        valid = self._valid(alpha)
        # Reverse traversal preserves original compositing order and lets each
        # leave-one-out comparison include all removals/replacements so far.
        prefix = (1 - alpha).cumprod(0)
        suffix = torch.ones_like(self.target)
        kept = []
        removed = moved = 0
        for i in reversed(range(len(strokes))):
            others = suffix * (prefix[i - 1] if i else 1)
            old_cost = self._cost(others * (1 - alpha[i]))
            removed_cost = self._cost(others)
            redundant = torch.max(others * alpha[i]) < .01
            unhelpful = removed_cost < old_cost - self.improvement_epsilon
            if not valid[i] or unhelpful or (not relocate and redundant):
                proposal = self._propose(painting, strokes[i], others) if relocate else None
                if proposal is not None and proposal[2] < old_cost - self.improvement_epsilon:
                    candidate, new_alpha, _ = proposal
                    self._copy_geometry(strokes[i], candidate)
                    # Momentum belongs to the old position, not its replacement.
                    for optimizer in optimizers:
                        if optimizer is not None:
                            for parameter in strokes[i].parameters():
                                optimizer.state.pop(parameter, None)
                    suffix *= 1 - new_alpha
                    kept.append(strokes[i])
                    moved += 1
                elif not valid[i] or removed_cost <= old_cost + self.improvement_epsilon:
                    removed += 1
                else:
                    suffix *= 1 - alpha[i]
                    kept.append(strokes[i])
            else:
                suffix *= 1 - alpha[i]
                kept.append(strokes[i])
        painting.brush_strokes = nn.ModuleList(reversed(kept))
        self.stats['relocated_strokes'] += moved
        self.stats['removed_strokes'] += removed
        self.stats['retained_strokes'] = len(kept)
        self.stats['refinements'].append(dict(step=step, relocated=moved, removed=removed, retained=len(kept)))
        if not kept:
            raise ValueError('No usable strokes remain for this silhouette')
