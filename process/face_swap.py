import numpy as np
import cv2

from model.detect_swap.PRNet.utils.render import render_texture
from objects.mask import Mask

from TMTChatbot import BaseSingleton


class FaceSwap(BaseSingleton):
    """
    This class for swapping faces
    """
    def __init__(self, swap_model):
        self.prn = swap_model
        self.render_texture = render_texture

    def rendered_img(self, mask: Mask):
        """
        Returns:
            np.ndarray: result of face swapping
        """
        mask_features = mask.mask_features
        new_colors = self.prn.get_colors_from_texture(mask.texture)
        new_img = self.render_texture(mask_features[0].T, 
                                   new_colors.T, 
                                   self.prn.triangles.T, 
                                   mask_features[1], 
                                   mask_features[2], 
                                   c=3)
        output = (new_img*255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def __call__(self, mask: Mask):
        try:
            mask.render_img = self.rendered_img(mask=mask)
        except:
            mask.render_img = None
        return mask
