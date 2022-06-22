from recognition.recognizer import FaceRecognizer

if __name__ == '__main__':
    recognizer = FaceRecognizer(train=True)
    recognizer.train(epochs=1)