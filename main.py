import os
import numpy as np
import cv2
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.utils import platform

M = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

AGES = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
GENDERS = ['Male', 'Female']
EMOTIONS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']
OBJECTS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
           "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
           "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
MEAN = (78.4263377603, 87.7689143744, 114.895847746)

BACK, FRONT = 0, 1


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
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 > 20 and y2 - y1 > 20:
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
        out = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf > 0.5:
                name = OBJECTS[int(det[0, 0, i, 1])]
                x1, y1, x2, y2 = (det[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
                out.append((name, conf, (max(0, x1), max(0, y1), min(w, x2), min(h, y2))))
        return out

    def analyze(self, img):
        """Returns (text, [(box, label), ...])."""
        boxes = self.faces(img)
        if boxes:
            lines, items = [], []
            for n, b in enumerate(boxes, 1):
                g, a, e = self.person(img, b)
                lines.append(f"Person {n}: {g}, age {a}, {e}")
                items.append((b, f"{g}, {a}, {e}"))
            return "\n".join(lines), items
        objs = self.objects(img)
        if objs:
            text = "Object(s): " + ", ".join(f"{n} ({c:.0%})" for n, c, _ in objs)
            return text, [(b, f"{n} {c:.0%}") for n, c, b in objs]
        return "Nothing recognised. Try better light.", []


class CamView(Image):
    """Live camera preview. Back camera by default, flip + rotate on demand."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.index = BACK
        self.rot = {BACK: 3, FRONT: 1}     # np.rot90 steps per camera; Rotate button cycles it
        self.cam = None
        self.frame = None                   # upright BGR frame without drawings
        self.overlay = []                   # [(box, label)]
        self.frozen = False
        self._open(self.index)
        Clock.schedule_interval(self.update, 1 / 20)

    def _open(self, index):
        self.index = index
        self.cam = Camera(index=index, resolution=(640, 480), play=True)

    def flip(self):
        if self.cam is not None:
            self.cam.play = False
            self.cam = None
        self.frame, self.overlay, self.frozen = None, [], False
        nxt = FRONT if self.index == BACK else BACK
        Clock.schedule_once(lambda dt: self._open(nxt), 0.4)

    def rotate(self):
        self.rot[self.index] = (self.rot[self.index] + 1) % 4

    def update(self, dt):
        if self.frozen or self.cam is None or self.cam.texture is None:
            return
        tex = self.cam.texture
        arr = np.frombuffer(tex.pixels, np.uint8).reshape(tex.height, tex.width, 4)
        top = np.rot90(arr[::-1], self.rot[self.index])   # arr[::-1] = natural top-down
        if self.index == FRONT:                            # mirror selfie only
            top = top[:, ::-1]
        self.frame = cv2.cvtColor(np.ascontiguousarray(top), cv2.COLOR_RGBA2BGR)
        self.show()

    def show(self):
        if self.frame is None:
            return
        img = self.frame.copy()
        for (x1, y1, x2, y2), label in self.overlay:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(img, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 220, 0), 2)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt="rgb")
        t.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        t.flip_vertical()
        self.texture = t


class Root(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", **kw)
        self.view = CamView()
        self.label = Label(text="Loading models...", size_hint_y=0.18,
                           halign="center", font_size="17sp")
        bar = BoxLayout(size_hint_y=0.12, spacing=4)
        self.b_flip = Button(text="Flip\ncamera")
        self.b_rot = Button(text="Rotate")
        self.b_live = ToggleButton(text="Live\nOFF")
        self.b_cap = Button(text="Capture")
        self.b_flip.bind(on_release=lambda *_: self.view.flip())
        self.b_rot.bind(on_release=lambda *_: self.view.rotate())
        self.b_live.bind(on_release=self.toggle_live)
        self.b_cap.bind(on_release=self.capture)
        for b in (self.b_flip, self.b_rot, self.b_live, self.b_cap):
            bar.add_widget(b)
        for w in (self.view, self.label, bar):
            self.add_widget(w)
        self.an = None
        Clock.schedule_once(self.load_models, 0.5)
        Clock.schedule_interval(self.live_tick, 1.0)

    def load_models(self, dt):
        try:
            self.an = Analyzer()
            self.label.text = "Ready. Tap Capture, or turn Live ON."
        except Exception as ex:
            self.label.text = f"Model error: {ex}"

    def toggle_live(self, *_):
        on = self.b_live.state == "down"
        self.b_live.text = "Live\nON" if on else "Live\nOFF"
        if not on:
            self.view.overlay = []

    def live_tick(self, dt):
        if self.b_live.state != "down" or self.an is None or self.view.frozen:
            return
        if self.view.frame is None:
            return
        self.run(self.view.frame)

    def run(self, img):
        try:
            text, items = self.an.analyze(img)
            self.view.overlay = items
            self.label.text = text
        except Exception as ex:
            self.label.text = f"Error: {ex}"

    def capture(self, *_):
        v = self.view
        if v.frozen:                       # Retake
            v.frozen, v.overlay = False, []
            self.b_cap.text = "Capture"
            self.label.text = "Ready."
            return
        if self.an is None or v.frame is None:
            self.label.text = "Not ready yet"
            return
        self.run(v.frame)
        v.frozen = True
        v.show()
        self.b_cap.text = "Retake"


class SelfieApp(App):
    def build(self):
        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA])
        return Root()


if __name__ == "__main__":
    SelfieApp().run()
