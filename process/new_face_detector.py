import numpy as np
import mediapipe as mp
import cv2

from process.utils import resize_img

from TMTChatbot import BaseSingleton


class FaceDetMesh(BaseSingleton):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8 
        )

        
# class NewFaceDetector:
#     def __init__(self, img: np.ndarray):
#         self.face_mesh = FaceDetMesh().face_mesh
#         self.img = img
#         self.img_H, self.img_W = img.shape[:2]
#         self._face_landmarks = None
#         self._cropped_bbox = None
#         self._expanded_cropped_bbox = None
#         self._cropped_face = None
#         self._lmk5 = None
        
#     @property
#     def face_landmarks(self):
#         if self._face_landmarks is None:
#             self._face_landmarks = self.face_mesh.process(self.img.copy())
#             self._face_landmarks = self._face_landmarks.multi_face_landmarks
#         return self._face_landmarks
    
#     def check_face_landmarks(self):
#         return bool(self.face_landmarks)

#     @staticmethod
#     def find_central_point(points: list):
#         points = np.array(points)
#         x1 = points[:, 0].min()
#         x2 = points[:, 0].max()
#         y1 = points[:, 1].min()
#         y2 = points[:, 1].max()
#         return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
#     @staticmethod
#     def expand_box(img: np.ndarray, box: list):
#         """
#         this func for expand area near the face if face found
#         Args:
#             img (np.ndarray): original image 
#             box (list): coordinate of face_crop

#         Returns:
#             np.ndarray: new coordinate
#         """
#         h, w = img.shape[:2]
#         box_w = box[2] - box[0]
#         box_h = box[3] - box[1]
#         new_x1 = max(int(box[0] - 2*box_w/5), 0)
#         new_x2 = min(int(box[2] + 2*box_w/5), w)
#         new_y1 = max(int(box[1] - 2*box_h/5), 0)
#         new_y2 = min(int(box[3] + 2*box_h/5), h)
#         return [new_x1, new_y1, new_x2, new_y2]  

#     @property
#     def cropped_bbox(self):
#         if self._cropped_bbox is None:
#             if self.check_face_landmarks():
#                 list_points_xy = []
#                 for point in self.face_landmarks[0].landmark:
#                     list_points_xy.append([int(point.x * self.img_W), int(point.y * self.img_H)])
#                 list_points_xy = np.array(list_points_xy)
#                 x1 = max(list_points_xy[:, 0].min(), 0)
#                 x2 = min(list_points_xy[:, 0].max(), self.img_W)
#                 y1 = max(list_points_xy[:, 1].min(), 0)
#                 y2 = min(list_points_xy[:, 1].max(), self.img_H)
#                 self._cropped_bbox = [x1, y1, x2, y2]
#             else:
#                 self._cropped_bbox = 0
#         return self._cropped_bbox
    
#     @property
#     def expanded_cropped_bbox(self):
#         if self._expanded_cropped_bbox is None:
#             if self.cropped_bbox != 0:
#                 self._expanded_cropped_bbox = self.expand_box(self.img, self.cropped_bbox)
#             else:
#                 self._expanded_cropped_bbox = 0
#         return self._expanded_cropped_bbox
    
#     @property
#     def cropped_face(self):
#         """
#         Returns:
#             np.ndarray: new face_crop are if face found else ori image
#         """
#         if self._cropped_face is None:
#             if self.expanded_cropped_bbox != 0:
#                 # self._cropped_face = self.img[self.expanded_cropped_bbox[1]:self.expanded_cropped_bbox[3], 
#                 #                               self.expanded_cropped_bbox[0]:self.expanded_cropped_bbox[2]]
#                 self._cropped_face = self.img.copy()
#             else:
#                 self._cropped_face = self.img.copy()
#         return self._cropped_face

#     @property
#     def lmk5(self):
#         if self._lmk5 is None:
#             if self.face_landmarks is not None:
#                 left_eye = [int((self.face_landmarks[0].landmark[159].x + self.face_landmarks[0].landmark[145].x) / 2*self.img_W), 
#                 int((self.face_landmarks[0].landmark[159].y + self.face_landmarks[0].landmark[145].y) / 2*self.img_H)]

#                 right_eye = [int((self.face_landmarks[0].landmark[386].x + self.face_landmarks[0].landmark[374].x) / 2*self.img_W), 
#                 int((self.face_landmarks[0].landmark[386].y + self.face_landmarks[0].landmark[374].y) / 2*self.img_H)]

#                 nose = [int((self.face_landmarks[0].landmark[19].x) * self.img_W), int((self.face_landmarks[0].landmark[19].y) * self.img_H)]

#                 left_mouth = [int((self.face_landmarks[0].landmark[61].x) * self.img_W), int((self.face_landmarks[0].landmark[61].y) * self.img_H)]

#                 right_mouth = [int((self.face_landmarks[0].landmark[291].x) * self.img_W), int((self.face_landmarks[0].landmark[291].y) * self.img_H)]

#                 self._lmk5 = [left_eye, right_eye, nose, left_mouth, right_mouth]

#                 # for id, point in enumerate(self._lmk5):
#                 #     self._lmk5[id] = [point[0] - self.expanded_cropped_bbox[0], point[1] - self.expanded_cropped_bbox[1]]
#                 self._lmk5 = np.array(self._lmk5)
#             else:
#                 self._lmk5 = 0
#         return self._lmk5


class NewFaceDetector:
    def __init__(self, img: np.ndarray):
        self.face_mesh = FaceDetMesh().face_mesh
        self.img = img
        self.img_H, self.img_W = img.shape[:2]
        self._face_landmarks = None
        self._cropped_bbox = None
        self._square_bbox = None
        self._expanded_cropped_bbox = None
        self._cropped_face = None
        self._lmk5 = None
        self.output_size = 256

    @property
    def face_landmarks(self):
        if self._face_landmarks is None:
            self._face_landmarks = self.face_mesh.process(self.img.copy())
            self._face_landmarks = self._face_landmarks.multi_face_landmarks
        return self._face_landmarks
    
    def check_face_landmarks(self):
        return bool(self.face_landmarks)

    @property
    def cropped_bbox(self):
        if self._cropped_bbox is None:
            if self.check_face_landmarks():
                list_points_xy = []
                for point in self.face_landmarks[0].landmark:
                    list_points_xy.append([int(np.floor(point.x * self.img_W)), int(np.floor(point.y * self.img_H))])
                list_points_xy = np.array(list_points_xy)
                x_list = list_points_xy[:, 0]
                y_list = list_points_xy[:, 1]
                x1 = x_list.min()
                x2 = x_list.max()
                y1 = y_list.min()
                y2 = y_list.max()
                self._cropped_bbox = [x1, y1, x2, y2]
            else:
                self._cropped_bbox = 0
        return self._cropped_bbox
    
    @property
    def square_bbox(self):
        if self._square_bbox is None:
            if self.cropped_bbox != 0:
                x1, y1, x2, y2 = self.cropped_bbox
                width = x2 - x1
                height = y2 - y1
                center_point = (int(np.floor(x1 + width/2)), int(np.floor(y1 + height/2)))
                if width > height:
                    std_size = width
                    y1 = int(np.floor(center_point[1] - std_size/2))
                    y2 = y1 + std_size
                elif width < height:
                    std_size = height
                    x1 = int(np.floor(center_point[0] - std_size/2))
                    x2 = x1 + std_size
                self._square_bbox = [x1, y1, x2, y2]
            else:
                self._square_bbox = 0
        return self._square_bbox

    @staticmethod
    def expand_bbox(img: np.ndarray, bbox: list, coef_expand=0.15):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = bbox
        old_size = x2 - x1 
        border_width = min(int(np.floor(coef_expand*old_size)) - 1, x1, w - x2, y1, h - y2)
        new_x1 = x1 - border_width
        new_y1 = y1 - border_width
        new_x2 = x2 + border_width
        new_y2 = y2 + border_width
        return [new_x1, new_y1, new_x2, new_y2]

    @property
    def expanded_cropped_bbox(self):
        if self._expanded_cropped_bbox is None:
            if self.square_bbox != 0:
                self._expanded_cropped_bbox = self.expand_bbox(self.img, self.square_bbox)
            else:
                self._expanded_cropped_bbox = 0
        return self._expanded_cropped_bbox
    
    @property
    def cropped_face(self):
        """
        Returns:
            np.ndarray: new face_crop are if face found else ori image
        """
        if self._cropped_face is None:
            if self.expanded_cropped_bbox != 0:
                self._cropped_face = self.img[self.expanded_cropped_bbox[1]:self.expanded_cropped_bbox[3], 
                                              self.expanded_cropped_bbox[0]:self.expanded_cropped_bbox[2]]
                self._cropped_face = resize_img(img=self._cropped_face, new_h=self.output_size, square=True)
                # self._cropped_face = self.img.copy()
            else:
                self._cropped_face = self.img.copy()
        return self._cropped_face

    @property
    def lmk5(self):
        if self._lmk5 is None:
            if self.face_landmarks is not None:
                ratio = self.output_size / (self.expanded_cropped_bbox[2] - self.expanded_cropped_bbox[0])
                left_eye = [int((self.face_landmarks[0].landmark[159].x + self.face_landmarks[0].landmark[145].x) / 2*self.img_W), 
                int((self.face_landmarks[0].landmark[159].y + self.face_landmarks[0].landmark[145].y) / 2*self.img_H)]

                right_eye = [int((self.face_landmarks[0].landmark[386].x + self.face_landmarks[0].landmark[374].x) / 2*self.img_W), 
                int((self.face_landmarks[0].landmark[386].y + self.face_landmarks[0].landmark[374].y) / 2*self.img_H)]

                nose = [int((self.face_landmarks[0].landmark[19].x) * self.img_W), int((self.face_landmarks[0].landmark[19].y) * self.img_H)]

                left_mouth = [int((self.face_landmarks[0].landmark[61].x) * self.img_W), int((self.face_landmarks[0].landmark[61].y) * self.img_H)]

                right_mouth = [int((self.face_landmarks[0].landmark[291].x) * self.img_W), int((self.face_landmarks[0].landmark[291].y) * self.img_H)]

                self._lmk5 = [left_eye, right_eye, nose, left_mouth, right_mouth]

                for id, point in enumerate(self._lmk5):
                    self._lmk5[id] = [int(np.floor((point[0] - self.expanded_cropped_bbox[0])*ratio)), int(np.floor((point[1] - self.expanded_cropped_bbox[1])*ratio))]
                self._lmk5 = np.array(self._lmk5)
            else:
                self._lmk5 = 0
        return self._lmk5