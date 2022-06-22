import cv2
import torch
import numpy as np

from detection.models.RetinaFace import RetinaFace
from detection.models.configs import cfg_re50, cfg_mnet
from detection.utils.prior_box import PriorBox
from detection.utils.cpu_nms import py_cpu_nms
from detection.utils.box_utils import decode, decode_landm


class FaceDetector:
    def __init__(self, name='resnet', weight_path=None, max_resolution=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 confidence_threshold=0.95, top_k=5000, nms_threshold=0.4,
                 keep_top_k=750):

        model, config = None, None
        if name == 'mobilenet':
            config = cfg_mnet
            model = RetinaFace(cfg=config, phase='test')
            weight_path = 'detection/weights/mobilenet.pth'
        elif name == 'resnet':
            config = cfg_re50
            model = RetinaFace(cfg=config, phase='test')
            weight_path = 'detection/weights/resnet50_2022-04-13.pth'
        else:
            exit('Failed to create face detector. Model name can be either mobilenet or resnet')

        # MODEL SETTINGS
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.half()
        model.to(device).eval()
        self.model = model.half()
        self.device = device
        self.cfg = config

        # DETECTION SETTINGS
        self.thresh = confidence_threshold
        self.top_k = top_k
        self.nms_thresh = nms_threshold
        self.keep_top_k = keep_top_k
        self.max_resolution = max_resolution

    def detect(self, image_raw):
        image = np.float32(image_raw)
        image_width = np.max(image.shape[0:2])
        resize = float(self.max_resolution) / float(image_width)
        image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        image_height, image_width, _ = image.shape
        scale = torch.Tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
        image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0)
        image = image.half()
        image = image.to(self.device)
        scale = scale.to(self.device)

        with torch.no_grad():
            loc, conf, landmarks = self.model(image)

        priorbox = PriorBox(self.cfg, image_size=(image_height, image_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landmarks = decode_landm(landmarks.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2], image.shape[3], image.shape[2],
                               image.shape[3], image.shape[2]])
        scale1 = scale1.to(self.device)
        landmarks = landmarks * scale1 / resize
        landmarks = landmarks.cpu().numpy()

        # ignore low scores
        indx = np.where(scores > self.thresh)[0]
        boxes = boxes[indx]
        landmarks = landmarks[indx]
        scores = scores[indx]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        # Do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_thresh)
        dets = dets[keep, :]
        landmarks = landmarks[keep]
        dets = np.concatenate((dets, landmarks), axis=1)

        return dets