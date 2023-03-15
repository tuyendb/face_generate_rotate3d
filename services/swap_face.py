import threading
import logging

from common.queue_name import QName
from services.buffer_manager import BufferManager
from process.face_swap import FaceSwap
from objects.mask import Mask
from objects.image_list import ImageList
from config.config import Config

from TMTChatbot import BaseServiceSingleton


class SwapFace(BaseServiceSingleton):
    def __init__(self, swap_model, config: Config = None):
        super(SwapFace, self).__init__(config=config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.swap_model = swap_model
        self.buffer_manager = BufferManager(config=config)
        self.worker = threading.Thread(target=self.job, daemon=True)
        self.face_swap = FaceSwap(swap_model)

    def start(self):
        self.worker.start()
    
    def job(self):
        while True:
            try:
                image_list: ImageList = self.buffer_manager.get_data(queue_name=QName.MASK_TEXTURE_Q)
                if image_list.compulsory_flag: 
                    for image in image_list.forward_image_list:
                        for mask in image.masks:
                            if not mask.status:
                                mask = self.face_swap(mask=mask)
                                mask.status = True
                                self.buffer_manager.put_data(queue_name=QName.RENDER_IMAGE_Q, data=image_list)
                                break
                        else:
                            continue
                        break
                else:
                    self.buffer_manager.put_data(queue_name=QName.RENDER_IMAGE_Q, data=image_list)
            except Exception as e:
                self.logger.error(f"Swap face error | {e}")
    
    def join(self):
        [self.buffer_manager.qs[queue].join() for queue in self.buffer_manager.qs]
        