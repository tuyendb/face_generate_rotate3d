import numpy as np
from PIL import Image

from model.detect_swap.Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
from model.detect_swap.Deep3DFaceRecon_pytorch.util.util import tensor2im
from process.utils import read_data
from objects.image import InImage
from config.config import Config

from TMTChatbot import BaseSingleton


class MaskCre(BaseSingleton):
    def __init__(self, opt, recon_model, swap_model):
        self.opt = opt
        self.config = Config()
        self.recon_model = recon_model
        self.prn = swap_model

    def processed_data(self, img, lmk5):
        """
        Args:
            img (PIL.objects): input image
        Returns:
            data
        """
        lm3d_std = load_lm3d(self.opt.bfm_folder)
        img_tensor, lm_tensor = read_data(img, lm3d_std, to_tensor=True, lmk5=lmk5)
        img_numpy = np.array(img.copy())[:,:,::-1]
        return img_numpy, img_tensor, lm_tensor

    def render_mask(self, img_tensor, lm_tensor, rotation):
        data = {
            'imgs': img_tensor,
            'lm': lm_tensor,
            'rotation': rotation
        }
        self.recon_model.set_input(data)
        self.recon_model.test()
        visual = self.recon_model.get_result_visuals()
        mask = tensor2im(visual[0])[:,:,::-1]
        return mask

    def auto_render(self, img, rotation):
        cropped_face = img.expanded_cropped_face
        lmk5 = img.lmk5
        pil_cropped_face = Image.fromarray(cropped_face[:,:,::-1]).convert('RGB')
        try:
            _, img_tensor, lm_tensor = self.processed_data(pil_cropped_face, lmk5)
            mask = self.render_mask(img_tensor, lm_tensor, rotation)
        except Exception as e:
            print(f"Render error | {e}")
            mask = None
        return mask

    def __call__(self, img: InImage, rotation):
        mask = self.auto_render(img=img, rotation=rotation)
        return mask
