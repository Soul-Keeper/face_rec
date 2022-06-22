import cv2
import torch
import numpy as np


def faces_preprocessing(faces, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    faces = faces.permute(0, 3, 1, 2).float()
    faces = faces.div(255).to(device)
    mu = torch.as_tensor([.5, .5, .5], dtype=faces.dtype, device=device)
    faces[:].sub_(mu[:, None, None]).div_(mu[:, None, None])
    return faces


class FaceNormalizer:
    def __init__(self, face_size=(112, 112), norm_type=cv2.NORM_MINMAX):
        self.face_size = face_size
        self.norm_type = norm_type

    def normalize(self, image, detections):
        detections = list(map(int, detections))

        eye_center = ((detections[5] + detections[7])//2, (detections[6] + detections[8])//2)
        angle = np.degrees(np.arctan2((detections[8] - detections[6]), (detections[7] - detections[5])))
        M = cv2.getRotationMatrix2D(eye_center, angle, 1)
        rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        eye_distance = int(((detections[5] - detections[7])**2 + (detections[6] - detections[8])**2)**0.5)
        a_x = eye_center[0] - eye_distance if eye_center[0] - eye_distance > 0 else 0
        a_y = eye_center[1] - int(eye_distance*0.4) if eye_center[1] - int(eye_distance*0.4) > 0 else 0
        b_x = eye_center[0] + eye_distance if eye_center[0] + eye_distance < image.shape[1] else image.shape[1]
        b_y = eye_center[1] + int(eye_distance*1.6) if eye_center[1] + int(eye_distance*1.6) < image.shape[0] else image.shape[0]
        new_a = (a_x, a_y)
        new_b = (b_x, b_y)

        s = abs(b_x - a_x) * abs(b_y - a_y)

        if s > 0:
            cropped_image = rotated_image[new_a[1]:new_b[1], new_a[0]:new_b[0], :]
            resized_image = cv2.resize(cropped_image, self.face_size)
            # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
            normalized_image = cv2.normalize(resized_image, None, 0, 255, self.norm_type)
            return normalized_image
        else:
            return np.zeros([112, 112, 3], dtype=np.uint8)



