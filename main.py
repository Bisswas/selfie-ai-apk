import os
import numpy as np
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.utils import platform

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

AGES = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
GENDERS = ['Male', 'Female']
EMOTIONS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']
OBJECTS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
MEAN = (78.4263377603, 87.7689143744, 114.895847746)


def load(fn, *a):
    return fn(*[os.path.join(M, x) for x in a])


class Analyzer:
    def __init__(self):
        self.face = load(cv2.dnn.readNetFromCaffe, "face_deploy.prototxt",
                         "res10_300x300_ssd_iter_140000.caffemodel")
        self.age = load(cv2.dnn.readNetFromCaffe, "age_deploy.prototxt", "age_net.caffemodel")
        self.gender = load(cv2.dnn.readNetFromCaffe, "gender_deploy.prototxt", "gender_net.caffemodel")
        self.emotion = cv2.dnn.readNetFromONNX(os.path.join(M, "emotion-ferplus-8.onnx"))
        self.obj = load(cv2.dnn.readNetFromCaffe, "MobileNetSSD_deploy.prototxt",
                        "MobileNetSSD_deploy.caffemodel")

    def faces(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        self.face.setInput(blob)
        det = self.face.forward()
        out = []
        for i in range(det.shape[2]):
            if det[0, 0, i, 2] > 0.6:
                x1, y1, x2, y2 = (det[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                if x2 > x1 and y2 > y1:
                    out.append((x1, y1, x2, y2))
        return out

    def person(self, img, box):
        x1, y1, x2, y2 = box
        pad = 20
        face = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MEAN, swapRB=False)
        self.gender.setInput(blob)
        g = GENDERS[int(self.gender.forward()[0].argmax())]
        self.age.setInput(blob)
        a = AGES[int(self.age.forward()[0].argmax())]
        gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32).reshape(1, 1, 64, 64)
        self.emotion.setInput(gray)
        e = EMOTIONS[int(self.emotion.forward()[0].argmax())]
        return g, a, e

    def objects(self, img):
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        self.obj.setInput(blob)
        det = self.obj.forward()
        found = {}
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf > 0.5:
                name = OBJECTS[int(det[0, 0, i, 1])]
                found[name] = max(found.get(name, 0), conf)
        return found

    def run(self, img):
        boxes = self.faces(img)
        if boxes:
            lines = []
            for n, b in enumerate(boxes, 1):
                g, a, e = self.person(img, b)
                lines.append(f"Person {n}: {g}, age {a}, emotion {e}")
            return "\n".join(lines)
        objs = self.objects(img)
        if objs:
            return "Object(s): " + ", ".join(f"{k} ({v:.0%})" for k, v in objs.items())
        return "Nothing recognised. Try better light."


class Root(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", **kw)
        self.cam = Camera(index=1, resolution=(640, 480), play=True)  # 1 = front
        self.label = Label(text="Tap the button to analyse", size_hint_y=0.2,
                           halign="center", font_size="18sp")
        btn = Button(text="Take selfie & analyse", size_hint_y=0.12)
        btn.bind(on_release=self.capture)
        for w in (self.cam, self.label, btn):
            self.add_widget(w)
        self.an = None

    def capture(self, *_):
        tex = self.cam.texture
        if tex is None:
            self.label.text = "Camera not ready"
            return
        try:
            if self.an is None:
                self.an = Analyzer()
            arr = np.frombuffer(tex.pixels, np.uint8).reshape(tex.height, tex.width, 4)
            img = cv2.flip(cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR), 0)
            self.label.text = self.an.run(img)
        except Exception as ex:
            self.label.text = f"Error: {ex}"


class SelfieApp(App):
    def build(self):
        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA])
        return Root()


if __name__ == "__main__":
    SelfieApp().run()
