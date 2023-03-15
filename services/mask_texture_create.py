import threading
import logging
import numpy as np
import cv2

from services.buffer_manager import BufferManager
from common.queue_name import QName
from objects.image_list import ImageList
from objects.mask import Mask
from process.cre_mask_texture import MaskCre
from process.create_texture import CreateTexture
from config.config import Config

from TMTChatbot import BaseServiceSingleton


class MaskTextureCreate(BaseServiceSingleton):
    def __init__(self, option, recon_model, swap_model, config: Config = None):
        super(MaskTextureCreate, self).__init__(config=config)
        self.option = option
        self.recon_model = recon_model
        self.swap_model = swap_model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.buffer_manager = BufferManager(config=config)
        self.worker = threading.Thread(target=self.job, daemon=True)
        self.mask_create = MaskCre(
                        opt=self.option,
                        recon_model=self.recon_model,
                        swap_model=self.swap_model
                    )

    def start(self):
        self.worker.start()

    def join(self):
        [self.buffer_manager.qs[queue].join() for queue in self.buffer_manager.qs]

    def job(self):
        while True:
            try:
                image_list: ImageList = self.buffer_manager.get_data(queue_name=QName.IMAGE_LIST_Q)
                angle_list = np.arange(
                    -image_list.desired_angle, 
                    image_list.desired_angle + 1, 
                    image_list.step
                    )
                for img_id, image in enumerate(image_list.forward_image_list):
                    cropped_face = image.expanded_cropped_face
                    cre_texture = CreateTexture(cropped_face=cropped_face, swap_model=self.swap_model)
                    texture = cre_texture.cre_texture()
                    if texture is not None:
                        none_case = 0
                        for angle1 in angle_list:
                            for angle2 in angle_list:
                                for angle3 in angle_list:
                                    rotation = {
                                        "0": angle1,
                                        "1": angle2,
                                        "2": angle3
                                    }
                                    mask = self.mask_create(img=image, rotation=rotation)
                                    if mask is not None:
                                        mask_obj = Mask(mask=mask, 
                                                        texture=texture, 
                                                        mask_features=cre_texture.mask_features(mask=mask))
                                        image.masks.append(mask_obj)
                                        del mask_obj
                                        self.buffer_manager.put_data(queue_name=QName.MASK_TEXTURE_Q, data=image_list)
                                    else:
                                        none_case += 1
                        if none_case == len(angle_list)**3:
                            image_list.compulsory_flag = False
                            self.buffer_manager.put_data(queue_name=QName.MASK_TEXTURE_Q, data=image_list)
                    else:
                        image_list.compulsory_flag = False
                        self.buffer_manager.put_data(queue_name=QName.MASK_TEXTURE_Q, data=image_list)
                    del cre_texture
                    image_list.full_list = (img_id == len(image_list.forward_image_list) - 1)
            except Exception as e:
                self.logger.error(f"Create mask and texture error | {e}")
