"""Inspect saved collection batches without opening the robot or camera."""

import gzip
from pathlib import Path

import numpy as np


def inspect_stroke_library(directory):
    """Return the saved sample count and last batch ID, refusing incomplete data."""
    directory = Path(directory)
    parameters = {p.name.removeprefix('stroke_parameters'): p
                  for p in directory.glob('stroke_parameters*.npy')}
    images = {p.name.removeprefix('stroke_intensities'): p
              for p in directory.glob('stroke_intensities*.npy')}
    if parameters.keys() != images.keys():
        raise ValueError('Stroke library has an incomplete batch; preserve it and repair before resuming')
    count, last_batch = 0, 0
    for suffix, path in sorted(parameters.items()):
        batch_id = int(Path(suffix).stem)
        params = np.load(path, allow_pickle=False)
        with gzip.open(images[suffix], 'rb') as stream:
            strokes = np.load(stream, allow_pickle=False)
        if (params.ndim != 2 or params.shape[1] != 4 or not len(params)
                or strokes.ndim != 3 or len(strokes) != len(params)
                or not np.isfinite(params).all() or not np.isfinite(strokes).all()):
            raise ValueError(f'Invalid stroke-library batch: {path.name}')
        count += len(params)
        last_batch = max(last_batch, batch_id)
    return count, last_batch
