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
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.8
        )

class FaceDetector:
    def __init__(self, img: np.array):
        self.img = img
        self.img_H, self.img_W = img.shape[:2]
        self.face_mesh = FaceDetMesh().face_mesh
        self.face_detection = FaceDetMesh().face_detection
        self._face_results = None
        self._cropped_bbox = None
        self._cropped_face = None
        self._expanded_cropped_box = None
        self._expanded_cropped_face = None
        self._new_face_landmarks = None
        self._lmk5 = None
        self._angle = None
        self.epa_crop_img_H, self.epa_crop_img_W = None, None

    @property
    def face_results(self):
        if self._face_results is None:
            self._face_results = self.face_detection.process(self.img.copy())
        return self._face_results

    def check_face_detection(self):
        return bool(self.face_results.detections)

    @property
    def cropped_bbox(self):
        if self._cropped_bbox is None:
            if len(self.face_results.detections) == 0:
                pass
            elif len(self.face_results.detections) == 1:
                face = self.face_results.detections[0]
                face_data = face.location_data
                bbox = face_data.relative_bounding_box
                new_x = int(bbox.xmin * self.img_W)
                new_y = int(bbox.ymin * self.img_H)
                new_w = int(bbox.width * self.img_W)
                new_h = int(bbox.height * self.img_H)  
                self._cropped_bbox = [new_x, new_y, new_w, new_h]
            else:
                score_list = []
                for _, face in enumerate(self.face_results.detections):
                    score_list.append(round(face.score[0], 2))
                    argmax_id = score_list.index(max(score_list))
                chosen_face = self.face_results.detections[argmax_id]
                face_data = chosen_face.location_data
                bbox = face_data.relative_bounding_box
                new_x = int(bbox.xmin * self.img_W)
                new_y = int(bbox.ymin * self.img_H)
                new_w = int(bbox.width * self.img_W)
                new_h = int(bbox.height * self.img_H)  
                self._cropped_bbox = [new_x, new_y, new_w, new_h]
        return self._cropped_bbox

    def expand_box(self, box: list):
        # return [x1, y1, x2, y2]
        new_x1 = max(int(box[0] - 1.5*box[2]/6), 0)
        new_x2 = min(int(box[0] + box[2] + 1.5*box[2]/6), self.img_W)
        new_y1 = max(int(box[1] - 2*box[3]/6), 0)
        new_y2 = min(int(box[1] + box[3] + box[3]/6), self.img_H)
        return [new_x1, new_y1, new_x2, new_y2] 

    @property
    def expanded_cropped_box(self):
        if self._expanded_cropped_box is None:
            self._expanded_cropped_box = self.expand_box(self.cropped_bbox)
        return self._expanded_cropped_box
    
    @property
    def cropped_face(self):
        #original crop
        if self._cropped_face is None:
            self._cropped_face = self.img.copy()[
                self.cropped_bbox[1] : self.cropped_bbox[1] + self.cropped_bbox[3],
                self.cropped_bbox[0] : self.cropped_bbox[0] + self.cropped_bbox[2]
            ]
            self._cropped_face = resize_img(img=self._cropped_face, new_h=256, square=True)
        return self._cropped_face

    @property
    def expanded_cropped_face(self):
        # expanded crop
        if self._expanded_cropped_face is None:
            self._expanded_cropped_face = self.img.copy()[
                self.expanded_cropped_box[1] : self.expanded_cropped_box[3],
                self.expanded_cropped_box[0] : self.expanded_cropped_box[2]
            ]
            self._expanded_cropped_face = resize_img(img=self._expanded_cropped_face, new_h=400, square=False)
            self.epa_crop_img_H, self.epa_crop_img_W = self._expanded_cropped_face.shape[:2]
        return self._expanded_cropped_face
    
    @property
    def new_face_landmarks(self):
        # get landmarks on expanded_cropped_face
        if self._new_face_landmarks is None:
            landmarks_results = self.face_mesh.process(self.expanded_cropped_face.copy())
            if landmarks_results.multi_face_landmarks:
                self._new_face_landmarks = landmarks_results.multi_face_landmarks
            else:
                self._new_face_landmarks = 0
        return self._new_face_landmarks

    def check_face(self):
        return self.check_face_detection() and self.new_face_landmarks != 0

    @property
    def lmk5(self):
        if self._lmk5 is None:
            if self.new_face_landmarks != 0:
                left_eye = [int((self.new_face_landmarks[0].landmark[159].x + self.new_face_landmarks[0].landmark[145].x) / 2*self.epa_crop_img_W), 
                int((self.new_face_landmarks[0].landmark[159].y + self.new_face_landmarks[0].landmark[145].y) / 2*self.epa_crop_img_H)]

                right_eye = [int((self.new_face_landmarks[0].landmark[386].x + self.new_face_landmarks[0].landmark[374].x) / 2*self.epa_crop_img_W), 
                int((self.new_face_landmarks[0].landmark[386].y + self.new_face_landmarks[0].landmark[374].y) / 2*self.epa_crop_img_H)]

                nose = [int((self.new_face_landmarks[0].landmark[19].x) * self.epa_crop_img_W), int((self.new_face_landmarks[0].landmark[19].y) * self.epa_crop_img_H)]

                left_mouth = [int((self.new_face_landmarks[0].landmark[61].x) * self.epa_crop_img_W), int((self.new_face_landmarks[0].landmark[61].y) * self.epa_crop_img_H)]

                right_mouth = [int((self.new_face_landmarks[0].landmark[291].x) * self.epa_crop_img_W), int((self.new_face_landmarks[0].landmark[291].y) * self.epa_crop_img_H)]

                self._lmk5 = [left_eye, right_eye, nose, left_mouth, right_mouth]
                self._lmk5 = np.array(self._lmk5)
            else:
                self._lmk5 = 0
        return self._lmk5
        
    @property
    def angle(self):
        if self._angle is None:
            if self.new_face_landmarks != 0:
                points_3d_choose = []
                points_2d_choose = []
                face_3d = []
                face_3d_viz = []
                for face_landmarks in self.new_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        face_3d.append([lm.x * self.epa_crop_img_W, lm.y * self.epa_crop_img_H, lm.z * 3000])
                        if idx in np.unique(list(mp.solutions.face_mesh.FACEMESH_LIPS)).tolist() + [1, 4, 5, 195, 197, 6, 2, 164]:    
                            x, y = int(lm.x * self.epa_crop_img_W), int(lm.y * self.epa_crop_img_H)
                            # Get the 2D Coordinates
                            points_2d_choose.append([x, y])
                            # Get the 3D Coordinates
                            points_3d_choose.append([x, y, lm.z])
                        else:
                            face_3d_viz.append([lm.x * self.epa_crop_img_W, lm.y * self.epa_crop_img_H, lm.z * 3000])
                    face_3d = np.array(face_3d)
                    face_3d_viz = np.array(face_3d_viz)
                    central_points = np.array([np.sum(face_3d[:,0]) / len(face_3d), np.sum(face_3d[:,1]) / len(face_3d), np.sum(face_3d[:,2]) / len(face_3d)])
                    # Convert it to the NumPy array
                    points_2d_choose = np.array(points_2d_choose, dtype=np.float64)
                    # Convert it to the NumPy array
                    points_3d_choose = np.array(points_3d_choose, dtype=np.float64)
                    # The camera matrix
                    focal_length = 1 * self.epa_crop_img_W

                    # cam_matrix = np.array([ [focal_length, 0, self.epa_crop_img_H/2],
                    #                         [0, focal_length, self.epa_crop_img_W/2],
                    #                         [0, 0, 1]])
                    cam_matrix = np.array([ [focal_length, 0, central_points[0]],
                                            [0, focal_length, central_points[1]],
                                            [0, 0, 1]])
                    # The distortion parameters
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(points_3d_choose, points_2d_choose, cam_matrix, dist_matrix)
                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)
                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                    # Get the rotation degree
                    x = angles[0] * 360 + 7.5
                    y = angles[1] * 360
                    z = angles[2] * 360
                self._angle = [-x, y, z]
            else: self._angle = 0
        return self._angle   

    def check_direction(self, angle):
        if angle == 0:
            return "Angle not found"
        else:
            x = -angle[0]
            y, z = angle[1:]
            epsilon = 5
            direction_list = []
            if (7 < x < 40):
                direction_list.append("Up")
            if (-40 < x < -7):
                direction_list.append("Down")
            if (10 < y < 40):
                direction_list.append("Right")
            if (-40 < y < -10):
                direction_list.append("Left")
            if (-10 <= y <= 10 and -10 <= x <= 10):
                direction_list.append("Forward")

            if (len(direction_list) == 0 or abs(x) >= 40 or abs(y) >= 40):
                return "Error"
            elif (len(direction_list) == 1):
                return direction_list[0]
            elif (len(direction_list) == 2):
                if (abs(x) > 20 and abs(y) > 20):
                    return "Error"
                if set(direction_list) == set(["Up", "Forward"]):
                    return "Up"
                if set(direction_list) == set(["Down", "Forward"]):
                    return "Down"
                if set(direction_list) == set(["Up", "Left"]):
                    if (abs(x) - abs(y) > epsilon):
                        return "Up"
                    return "Left"
                if set(direction_list) == set(["Down", "Left"]):
                    if (abs(x) - abs(y) > epsilon):
                        return "Down"
                    return "Left"
                if set(direction_list) == set(["Up", "Right"]):
                    if (abs(x) - abs(y) > epsilon):
                        return "Up"
                    return "Right"
                if set(direction_list) == set(["Down", "Right"]):
                    if (abs(x) - abs(y) > epsilon):
                        return "Down"
                    return "Right"
                
            elif (len(direction_list) == 3):
                if ("Left" in direction_list):
                    return "Left"
                if ("Right" in direction_list):
                    return "Right"

    def is_forward(self):
        #face detection, face landmarks detection and forward
        return self.check_face() and self.check_direction(self.angle) == "Forward"
        