import os
import cv2
import torch
import numpy as np

from pathlib import Path

from detection.detector import FaceDetector
from alignment.alignment import FaceNormalizer, faces_preprocessing
from recognition.recognizer import FaceRecognizer
from recognition.models.IR import l2_norm


def create_facebank(facebank_name):
    facebank_path = Path('recognition/banks/' + str(facebank_name).lower())
    if not facebank_path.exists():
        facebank_path.mkdir()


def add_person(facebank_name, person_name, images_path):
    facebank_person_path = Path('recognition/banks/' + str(facebank_name).lower() + '/' + str(person_name).lower())
    if not facebank_person_path.exists():
        facebank_person_path.mkdir()

    images_path = Path(images_path)
    if not images_path.exists():
        print("Path to folder with person's images doesn't exist")

    detector = FaceDetector()
    normalizer = FaceNormalizer()

    counter = 0
    for image_path in images_path.iterdir():
        image = cv2.imread(str(image_path))
        detections = detector.detect(image)
        face = normalizer.normalize(image=image, detections=detections[0])
        if len(face.shape) > 1:
            save_path = str(facebank_person_path) + '/' + str(counter) + '.jpg'
            cv2.imwrite(save_path, face)
            counter += 1
        else:
            print("There's no any faces in image: {}".format(image_path))
    print("{} successfully added to {}".format(person_name, facebank_name))

    update_facebank_pth('recognition/banks/' + facebank_name)


def update_facebank_pth(facebank_path, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    facebank_path = Path(facebank_path)
    if not facebank_path.exists():
        print("facebank doesn't exist")

    recognizer = FaceRecognizer()
    recognizer.model.eval()

    embeddings = torch.empty(0).to(device)
    names = np.array(['Unknown'])

    for path in facebank_path.iterdir():
        if path.is_file():
            continue
        faces = []
        for file in path.iterdir():
            face = cv2.imread(str(file))
            face = torch.tensor(face).unsqueeze(0)
            faces.append(face)

        faces = torch.cat(faces)
        if len(faces.shape) <= 3:
            continue

        with torch.no_grad():
            faces = faces_preprocessing(faces)
            faces_embeddings = recognizer.model(faces)
            flipped_faces_embeddings = recognizer.model(faces.flip(-1))
            final_embeddings = l2_norm(faces_embeddings + flipped_faces_embeddings)

        embeddings = torch.cat((embeddings, final_embeddings.mean(0, keepdim=True)))
        names = np.append(names, path.name)

    torch.save(embeddings, str(facebank_path) + '/embeddings.pth')
    np.save(str(facebank_path) + '/names', names)
    print('embeddings for this bank successfully updated')


def load_facebank_pth(facebank_name):
    facebank_path = Path('recognition/banks/' + facebank_name)
    if not facebank_path.exists():
        print("facebank doesn't exist")

    embeddings = torch.load(str(facebank_path) + '/embeddings.pth')
    names = np.load(str(facebank_path) + '/names.npy')
    return embeddings, names


def draw_bbox(image, detections, draw_landmarks=False, person_name=None, person_score=None, print_detection_score=False):
    b = list(map(int, detections))
    cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (125, 0, 255), 2)

    if print_detection_score:
        text = "{:.4f}".format(b[4])
        cv2.putText(image, text, (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    if (person_name is not None) and (person_score is not None):
        cv2.putText(image, str(person_name) + ' || ' + str(person_score), (b[0], b[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    if draw_landmarks:
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 7)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 7)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 7)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 7)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 7)

    return image





