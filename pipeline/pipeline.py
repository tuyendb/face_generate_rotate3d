from config.config import Config
from services.mask_texture_create import MaskTextureCreate
from services.swap_face import SwapFace
from services.upload_image import UploadImage


class Pipeline:
    def __init__(self, option, recon_model, swap_model):
        self.option = option
        self.config = Config()
        self.mask_texture_cre = MaskTextureCreate(option, recon_model, swap_model, config=self.config)
        self.swap_face = SwapFace(swap_model, config=self.config)
        self.upload_image = UploadImage(config=self.config)

    def start(self):
        self.mask_texture_cre.start()
        self.swap_face.start()
        self.upload_image.start()

    def join(self):
        self.mask_texture_cre.join()
        self.swap_face.join()
        self.upload_image.join()
