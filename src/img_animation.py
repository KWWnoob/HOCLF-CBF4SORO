__all__ = ["animate_images_cv2"]
import cv2  # importing cv2
from jax import Array
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as onp
import os
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Union


def animate_images_cv2(
    ts: onp.ndarray,
    img_ts: onp.ndarray,
    filepath: os.PathLike,
    speed_up: Union[float, Array] = 1,
    skip_step: int = 1,
    rgb_to_bgr: bool = True,
    **kwargs,
):
    """
    Animate using OpenCV
    Args:
        ts: time steps of the data
        img_ts: predicted images of shape (num_time_steps, width, height, channels)
        filepath: path to the output video
        speed_up: The speed up factor of the video.
        skip_step: The number of time steps to skip between animation frames.
        rgb_to_bgr: whether to convert the images from RGB to BGR
        **kwargs: Additional keyword arguments for the rendering function.

    Returns:

    """
    # extract parameters
    dt = onp.mean(onp.diff(ts)).item()
    fps = float(speed_up / (skip_step * dt))

    # create video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(
        str(filepath),
        fourcc,
        fps,  # fps,
        tuple(img_ts.shape[1:3]),
    )

    # skip frames
    ts = ts[::skip_step]
    img_ts = img_ts[::skip_step]

    # convert to RBG if grayscale
    if img_ts.shape[-1] == 1:
        img_ts = onp.repeat(img_ts, 3, axis=-1)

    if rgb_to_bgr:
        img_ts = onp.stack(
            [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img_ts], axis=0
        )

    for time_idx, t in tqdm(enumerate(ts)):
        video.write(img_ts[time_idx])

    video.release()
