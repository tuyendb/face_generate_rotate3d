import cv2
import numpy as np
import base64
import torch
import requests

from common.common_keys import *
from model.detect_swap.Deep3DFaceRecon_pytorch.util.preprocess import align_img
from config.config import Config


def convert_bw_b64_np(input_type: str, input_value):
    image = None
    if input_type == BASE64_IMAGE:
        im_bytes = base64.b64decode(input_value)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
        image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    elif input_type == NUMPY_IMAGE:
        encoded_img = cv2.imencode('.jpg', input_value)[1]
        image = base64.b64encode(encoded_img.tobytes()).decode("utf-8")
    return image


def resize_img(img: np.array, new_h=256, square=False):
    if square:
        new_img = cv2.resize(img, (new_h, new_h))
    else:
        old_h, old_w = img.shape[:2]
        ratio = new_h/old_h
        new_w = int(ratio*old_w)
        new_img =  cv2.resize(img, (new_w, new_h))
    return new_img


def read_data(img, lm3d_std, to_tensor=True, lmk5=None):
    """
    Args:
        img (PIL img): input image
        lm3d_std (np.ndarray): 5 points 3d landmarks of face
        to_tensor (bool, optional): Defaults to True.

    Returns:
        torch.tensor: img, landmarks
    """
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    W, H = img.size
    # lm = lmk5(open_cv_image).astype(np.float32)
    lm = lmk5
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(img, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm
