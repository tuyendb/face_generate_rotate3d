import cv2
import numpy as np


class CreateTexture:
    def __init__(self, cropped_face, swap_model):
        self.cropped_face = cropped_face
        self.prn = swap_model

    def cre_texture(self):
        try:
            ref_img = cv2.cvtColor(self.cropped_face.copy(), cv2.COLOR_BGR2RGB)
            ref_pos = self.prn.process(ref_img)
            ref_img = ref_img/255.
            ref_texture = cv2.remap(ref_img, 
                                ref_pos[:,:,:2].astype(np.float32), 
                                None, 
                                interpolation=cv2.INTER_NEAREST, 
                                borderMode=cv2.BORDER_CONSTANT,borderValue=(0)
                                )
        except:
            ref_texture = None
        return ref_texture

    def mask_features(self, mask):
        """
        mask features are vertices, shape(h, w) 
        Returns:
            tuple: mask features
        """
        h, w = mask.shape[:2]
        mask = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2RGB)
        pos = self.prn.process(mask) 
        vertices = self.prn.get_vertices(pos)
        return (vertices, h, w)
