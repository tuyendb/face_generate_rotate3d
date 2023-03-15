from typing import List
import numpy as np
from io import BytesIO
import cv2
import aiohttp
import asyncio
import time

from objects.image import InImage
from common.common_keys import *
from process.utils import convert_bw_b64_np


class ImageList:
    def __init__(self,input_list: list, user_id: str = None, desired_angle: int = None, step : int = None):
        self.input_list = input_list
        self.user_id = user_id
        self.desired_angle = desired_angle
        self.step = step
        self._image_list = None
        self._forward_image_list = None
        self._render_image_list = None
        self.compulsory_flag = True
        self.full_list = False
        self.num_mask_processing = 0
        self._num_of_processing = None
        self._uploaded_url_list = None

    # async def get_decode_image_from_url(self, url):
    #     for _ in range(5):
    #         try:
    #             async with aiohttp.ClientSession() as session:
    #                 response = await session.get(url, timeout=30)
    #                 buffer = BytesIO(await response.read())
    #                 arr = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
    #                 img = cv2.imdecode(arr, -1)
    #                 return img
    #         except Exception as e:
    #             print(f"Get Url Error: {e}")

    # async def get_decode_image_from_list_url(self, sites):
    #     tasks = []
    #     for url in sites:
    #         task = asyncio.ensure_future(self.get_decode_image_from_url(url))
    #         tasks.append(task)
    #     img_from_list_url = await asyncio.gather(*tasks, return_exceptions=True)
    #     return img_from_list_url
        
    # @property
    # def image_list(self):
    #     if self._image_list is None:
    #         st = time.time()
    #         self._image_list = asyncio.new_event_loop().run_until_complete(self.get_decode_image_from_list_url(self.input_list))
    #         print("GET URL TIMEEEEEE", time.time() - st)
    #         if bool(self._image_list):
    #             remove_ls = []
    #             for val in self._image_list:
    #                 if val is not None:
    #                     remove_ls.append(val)
    #             self._image_list = remove_ls
    #             del(remove_ls)
    #         else:
    #             pass
    #     return self._image_list

                                        #### For Input: List of general url
    # @property
    # def forward_image_list(self):
    #     if self._forward_image_list is None:
    #         self._forward_image_list = []
    #         if bool(self.image_list):
    #             for i, image in enumerate(self.image_list):
    #                 img = InImage(image)
    #                 if img.is_forward():
    #                     self._forward_image_list.append(img)
    #         else:
    #             pass
    #     return self._forward_image_list

                                        #### For Input: List of forward face url
    # @property
    # def forward_image_list(self):
    #     if self._forward_image_list is None:
    #         self._forward_image_list = []
    #         if bool(self.image_list):
    #             for image in self.image_list:
    #                 self._forward_image_list.append(InImage(image))
    #                 print(self._forward_image_list)
    #     return self._forward_image_list

    @property
    def forward_image_list(self):
        if self._forward_image_list is None:
            self._forward_image_list = []
            for input_value in self.input_list:
                array_img = convert_bw_b64_np(input_type=BASE64_IMAGE, input_value=input_value)
                self._forward_image_list.append(InImage(array_img))
                print(self._forward_image_list)
        return self._forward_image_list
                
    def from_json(data):
        return ImageList(input_list=data[INPUT_LIST], 
                         user_id=data[ID],
                         desired_angle=data[DESIRED_ANGLE],
                         step=data[STEP]
                         )

    @property
    def render_image_list(self):
        if self._render_image_list is None:
            self._render_image_list = []
            for forward_image in self.forward_image_list:
                render_list = []
                for mask in forward_image.masks:
                    render_b64 = convert_bw_b64_np(input_type=NUMPY_IMAGE, input_value=mask.render_img)
                    render_list.append(render_b64)
                self._render_image_list.append(render_list)
        return self._render_image_list

    @property
    def num_of_processing(self):
        if self._num_of_processing is None:
            if self.full_list:
                self._num_of_processing = 0
                for image in self.forward_image_list:
                    self._num_of_processing += len(image.masks)
            else:
                self._num_of_processing = None
        return self._num_of_processing

    @property
    def uploaded_url_list(self):
        if self._uploaded_url_list is None:
            self._uploaded_url_list = []
            for forward_image in self.forward_image_list:
                render_list = []
                for mask in forward_image.masks:
                    render_list.append(mask.uploaded_url)
                self._uploaded_url_list.append(render_list)
        return self._uploaded_url_list

    def all_data(self, i):
        data = {
            "0": {
                "User_id": self.user_id,
                "Return_list": self.uploaded_url_list,
                "Check": "Forward Faces exist !",
                "Status": "Done"
            },
            "1": {
                "User_id": self.user_id,
                "Return_list": None,
                "Check": "No mask rendered",
                "Status": "Done"
            },
        }    
        return data[str(i)]
