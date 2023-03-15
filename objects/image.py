import numpy as np
from typing import List

# from process.face_detector import FaceDetector
from process.new_face_detector import NewFaceDetector
from process.utils import convert_bw_b64_np
from common.common_keys import *
from objects.mask import Mask


class InImage:
    def __init__(self, img: np.array, masks: List[Mask] = None):
        self.img = img
        self._array_img = None
        self._b64_str = None
        # self.face_detector = FaceDetector(img)
        self.face_detector = NewFaceDetector(img)
        self.masks = masks if masks is not None else []
        # self.full_list = False

    @property
    def b64_str(self):
        if self._b64_str is None:
            self._b64_str = convert_bw_b64_np(input_type=NUMPY_IMAGE, input_value=self.expanded_cropped_face)
        return self._b64_str

    @property
    def cropped_face(self):
        return self.face_detector.cropped_face

    @property
    def expanded_cropped_face(self):
        return self.face_detector.cropped_face

    # @property
    # def angle(self):
    #     return self.face_detector.angle

    @property
    def lmk5(self):
        return self.face_detector.lmk5
    
    # def is_forward(self):
    #     return self.face_detector.is_forward()
