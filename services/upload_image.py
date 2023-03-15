import threading
import cv2
from PIL import Image
from datetime import timezone
import requests
import hashlib
import logging
import time
import io

from TMTChatbot import BaseServiceSingleton

from config.config import Config
from services.buffer_manager import BufferManager
from objects.mask import Mask
from objects.image_list import ImageList
from common.common_keys import *
from common.queue_name import QName


class UploadImage(BaseServiceSingleton):
    def __init__(self, config: Config = None):
        super(UploadImage, self).__init__(config=config)
        self.buffer_manager = BufferManager(config=config)
        self.worker = threading.Thread(target=self.job, daemon=True)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def start(self):
        self.worker.start()

    def join(self):
        [self.buffer_manager.qs[queue].join() for queue in self.buffer_manager.qs]

    # def upload_image(self, mask: Mask):
    #     _, im_buf_arr = cv2.imencode(".jpg", mask.render_img)
    #     byte_im = im_buf_arr.tobytes()
    #     suffix = hashlib.md5(byte_im).hexdigest()
    #     payload = {'IsUseDefaultName': 'true'}
    #     name = "face_gen_{}.jpg".format(suffix)
    #     files = [('FileContent', (name, byte_im, 'image/jpeg'))]
    #     headers = {}
    #     response = None
    #     while response is None:
    #         response = requests.request("POST", url=self.config.file_server_url, headers=headers, data=payload, files=files)
    #         time.sleep(0.01)
    #     api_url = self.config.file_server_get_url + name
    #     return api_url

    def upload_image(self, mask: Mask):
        image = Image.fromarray(mask.render_img[:,:,::-1])
        headers = {
            "accept": "text/plain",
            "App-Id": "tpos.vn",
            "App-Secret": self.config.secret_key,
            "App-Folder": "tpos"
        }
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='jpeg')
        image = img_byte_arr.getvalue()
        output_name = str(int(hashlib.sha1(image).hexdigest(), 16) % (10 ** 8)) + ".jpg"
        files = {
            "Files": (f"face_gen_{output_name}.jpg",
                      image, 
                      "image/jpeg"),
            "FolderPath": (None, "team-ai"),
            "Overwrite": (None, "true"),
            "Tenant": (None, ""),
        }
        result = []
        for _ in range(self.config.max_retries):
            try:
                result = requests.post(self.config.fileserver_url, headers=headers,
                                    files=files, timeout=self.config.time_out).json()[0]
                break
            except Exception as e:
                print(f"Push image to fileserver Fail: {e}")
        image = result["urlImageProxy"] if "urlImageProxy" in result else "Server Error"
        return image

    def job(self):
        while True:
            try:
                image_list: ImageList = self.buffer_manager.get_data(queue_name=QName.RENDER_IMAGE_Q)
                if image_list.compulsory_flag:
                    for image in image_list.forward_image_list:
                        for mask in image.masks:
                            if mask.status and not mask.uploaded:
                                st = time.time()
                                mask.uploaded_url = self.upload_image(mask=mask)
                                print("UPLOAD TIMEEEE", time.time() - st)
                                mask.uploaded = True
                                image_list.num_mask_processing += 1
                                break
                        else:
                            continue
                        break
                    print("Check Done",image_list.full_list, image_list.num_of_processing, image_list.num_mask_processing)
                    if image_list.full_list:
                        if image_list.num_mask_processing == image_list.num_of_processing:
                            self.buffer_manager.put_data(queue_name=QName.OUTPUT_Q, data=image_list)
                            print("PUT DONE")
                else:
                    self.buffer_manager.put_data(queue_name=QName.OUTPUT_Q, data=image_list)
            except Exception as e:
                self.logger.error(f"Upload image error | {e}")
