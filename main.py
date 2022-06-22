import cv2
import time
import torch
import numpy as np

import psutil
from pynvml.smi import nvidia_smi

from detection.detector import FaceDetector
from alignment.alignment import FaceNormalizer
from recognition.recognizer import FaceRecognizer
from utils import create_facebank, add_person, load_facebank_pth, draw_bbox

print("CUDA: {}".format(torch.cuda.is_available()))
print("DEVICE: {}".format(torch.cuda.get_device_name(0)))

if __name__ == '__main__':
    nvsmi = nvidia_smi.getInstance()
    gpu_start = nvsmi.DeviceQuery('memory.free, memory.total')
    ram_start = psutil.virtual_memory()[4]

    detector = FaceDetector()
    normalizer = FaceNormalizer()
    recognizer = FaceRecognizer(weights_path='recognition/weights/mobilenet.pth')
    embeddings, names = load_facebank_pth('test')

    cap = cv2.VideoCapture("videos/test_alignment.mp4")
    num_frames = 0
    time_start = time.time()

    while True:
        ret, frame = cap.read()
        k = cv2.waitKey(1)
        if k % 256 == 27 or not ret:  # ESC
            break
        num_frames += 1

        detections = detector.detect(frame)

        faces = []
        for person_detections in detections:
            face = normalizer.normalize(image=frame, detections=person_detections)
            faces.append(face)

        if len(faces) < 1:
            continue
        faces = torch.tensor(np.array(faces)).to(detector.device)
        results, scores = recognizer.infer(faces, embeddings)

        if num_frames == 51:
            nvsmi = nvidia_smi.getInstance()
            gpu_loop = nvsmi.DeviceQuery('memory.free, memory.total')
            ram_loop = psutil.virtual_memory()[4]

        for idx, person_detections in enumerate(detections):
            draw_bbox(image=frame, detections=person_detections, person_name=names[results[idx] + 1], person_score=scores[idx])

        cv2.imshow('recognition_test', frame)

    loop_time = time.time() - time_start

    cap.release()
    cv2.destroyAllWindows()

    fps = num_frames / loop_time
    gpu_memory_usage = round(float(gpu_start['gpu'][0]['fb_memory_usage']['free']) - float(gpu_loop['gpu'][0]['fb_memory_usage']['free']), 3)
    ram_usage = round((ram_start - ram_loop) / 1024 / 1024, 3)

    print("fps: {} || ram_usage: {} || gpu_memory_usage: {} MiB".format(round(fps, 3), ram_usage, gpu_memory_usage))

