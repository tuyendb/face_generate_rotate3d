from process.utils import convert_bw_b64_np
from common.common_keys import *


class Mask:
    def __init__(self, mask, texture, mask_features):
        self.mask = mask
        self.texture = texture
        self.mask_features = mask_features
        self.status = False
        self.uploaded = False
        self.render_img = None 
        self.uploaded_url = None

    @property
    def b64_str(self):
        if self.render_img is not None:
            return convert_bw_b64_np(input_type=NUMPY_IMAGE, input_value=self.render_img)
        else:
            return None
