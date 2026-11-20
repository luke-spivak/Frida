# README media

These assets show the physical SO101 prototype.

| Asset | Source | Preparation |
| --- | --- | --- |
| `so101-painting.jpg` | `IMG_6418.jpg` | Metadata removed for public sharing; original image pixels preserved. |
| `painting-demo.gif` | `overhead20x.mp4` | Seconds 2–14, brightness/contrast/gamma adjusted, scaled to 512 × 288, 8 fps, 64 colors, looping. |

The GIF retains the source video's playback speed; the source is labeled 20×.
It shows an excerpt from a painting run, not the entire run. The photograph
and overhead footage show the prototype from different viewpoints; they are
not a synchronized pair.

To regenerate the GIF from the supplied video, run this from the repository
root, substituting the path to the source video:

```sh
ffmpeg -hide_banner -loglevel error \
  -ss 2 -t 12 -i /path/to/overhead20x.mp4 \
  -filter_complex '[0:v]fps=8,scale=512:-1:flags=lanczos,eq=brightness=0.04:contrast=1.15:gamma=1.5:saturation=0.85,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle' \
  -loop 0 docs/media/painting-demo.gif
```

The original videos are kept outside the repository.
