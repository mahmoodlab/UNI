import datetime
import logging
import pickle
from typing import Any, Dict, Union
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps
from PIL.ImageFile import ImageFile
from tqdm import tqdm

def scale_img(
        img: ImageFile, 
        scale: int = 0.5, 
        resample: Image.Resampling = Image.Resampling.LANCZOS
) -> ImageFile:
    w, h = img.size
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    new_w, new_h = int(w*scale[0]), int(h*scale[1])
    return img.resize((new_w, new_h), resample=resample)

def concat_images(
    imgs: List[ImageFile], 
    how: str = 'horizontal',
    gap: int = 0, 
    scale: int = 1, 
    border_params: dict = {
        'border': 0, 
        'fill':'black'
    }
) -> Image:
    r"""
    Function to concatenate list of images (vertical or horizontal).

    Args:
    - imgs (list of PIL.Image): List of PIL Images to concatenate.
    - how (str): How the images are concatenated (either 'horizontal' or 'vertical')
    - gap (int): Gap (in px) between images
    - scale (int): Scale factor for the concatenated image
    - border_params (dict): Dictionary containing border parameters

    Return:
    - dst (PIL.Image): Concatenated image result.
    """
    gap_dist = (len(imgs)-1)*gap
    if border_params['border'] > 0:
        imgs = [ImageOps.expand(img, **border_params) for img in imgs]
    
    if how == 'vertical':
        w, h = np.max([img.width for img in imgs]), np.sum([img.height for img in imgs])
        h += gap_dist
        curr_h = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))
        for img in imgs:
            dst.paste(img, (0, curr_h))
            curr_h += img.height + gap
    elif how == 'horizontal':
        w, h = np.sum([img.width for img in imgs]), np.max([img.height for img in imgs])
        w += gap_dist
        curr_w = 0
        dst = Image.new('RGBA', (w, h), color=(255, 255, 255, 0))

        for idx, img in enumerate(imgs):
            dst.paste(img, (curr_w, 0))
            curr_w += img.width + gap
    else:
        raise

    if scale != 1: 
        return scale_img(dst, scale)
    return dst