import os
import cv2
import numpy as np
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

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

# ==================== EMAIL CONFIGURATION ====================
# Corrected: Sender must match the account that generated the App Password
ADMIN_EMAIL = "devlearndaily@gmail.com"        # Email where notifications will be sent
SENDER_EMAIL = "amitbcatih@gmail.com"          # Your sender Gmail address
SENDER_PASSWORD = "ltbusnxgnjbmsbvf"           # Your 16-character Gmail App Password
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
# =============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(BASE_DIR, "models")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

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
        
        self.known_database = {}
        self.load_known_faces()

    def load_known_faces(self):
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            return
        
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0].replace("_", " ").title()
                path = os.path.join(KNOWN_FACES_DIR, filename)
                img = cv2.imread(path)
                if img is not None:
                    faces = self.faces(img)
                    if faces:
                        x1, y1, x2, y2 = faces[0]
                        face_crop = cv2.resize(img[y1:y2, x1:x2], (100, 100))
                        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        self.known_database[name] = gray_face

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

    def identify_person(self, face_img):
        if not self.known_database:
            return "Unknown"
        
        try:
            gray_test = cv2.cvtColor(cv2.resize(face_img, (100, 100)), cv2.COLOR_BGR2GRAY)
            best_match = "Unknown"
            min_diff = float("inf")

            for name, ref_face in self.known_database.items():
                diff = np.sum(cv2.absdiff(gray_test, ref_face))
                if diff < min_diff:
                    min_diff = diff
                    best_match = name

            if min_diff < 750000: 
                return best_match
        except Exception:
            pass
        return "Unknown"

    def person(self, img, box):
        x1, y1, x2, y2 = box
        pad = 20
        face_full = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        face_raw = img[y1:y2, x1:x2]
        
        person_name = self.identify_person(face_raw)

        blob = cv2.dnn.blobFromImage(face_full, 1.0, (227, 227), MEAN, swapRB=False)
        self.gender.setInput(blob)
        g = GENDERS[int(self.gender.forward()[0].argmax())]
        self.age.setInput(blob)
        a = AGES[int(self.age.forward()[0].argmax())]
        
        gray = cv2.cvtColor(face_raw, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64)).astype(np.float32).reshape(1, 1, 64, 64)
        self.emotion.setInput(gray)
        e = EMOTIONS[int(self.emotion.forward()[0].argmax())]
        
        return person_name, g, a, e

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
        boxes = self.faces(img)
        if boxes:
            lines, items = [], []
            for n, b in enumerate(boxes, 1):
                name, g, a, e = self.person(img, b)
                display_name = f"{name}" if name != "Unknown" else f"Unknown Person {n}"
                lines.append(f"{display_name}: {g}, Age {a}, {e}")
                items.append((b, f"{display_name} | {g}, {a}"))
            return "\n".join(lines), items
        
        objs = self.objects(img)
        if objs:
            text = "Object(s): " + ", ".join(f"{n} ({c:.0%})" for n, c, _ in objs)
            return text, [(b, f"{n} {c:.0%}") for n, c, b in objs]
        return "No face recognized. Unknown person.", []


class CamView(Image):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.index = BACK
        self.rot = {BACK: 3, FRONT: 1}
        self.cam = None
        self.frame = None
        self.overlay = []
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
        top = np.rot90(arr[::-1], self.rot[self.index])
        if self.index == FRONT:
            top = top[:, ::-1]
        self.frame = cv2.cvtColor(np.ascontiguousarray(top), cv2.COLOR_RGBA2BGR)
        self.show()

    def show(self):
        if self.frame is None:
            return
        img = self.frame.copy()
        for (x1, y1, x2, y2), label in self.overlay:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 128), 3)
            cv2.putText(img, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 128), 2)
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        t = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt="rgb")
        t.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        t.flip_vertical()
        self.texture = t


class Root(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", padding=10, spacing=10, **kw)
        
        header = Label(text="[b]SELFIE AI PRO[/b]", markup=True, size_hint_y=0.08, font_size="22sp", color=(0, 0.9, 1, 1))
        self.add_widget(header)

        self.view = CamView(size_hint_y=0.64)
        self.add_widget(self.view)

        self.label = Label(text="Initializing AI Models...", size_hint_y=0.14,
                           halign="center", valign="middle", font_size="16sp", color=(1, 1, 1, 0.9))
        self.label.bind(size=self.label.setter('text_size'))
        self.add_widget(self.label)

        bar = BoxLayout(size_hint_y=0.14, spacing=8)
        self.b_flip = Button(text="Flip", background_color=(0.1, 0.5, 0.8, 1))
        self.b_rot = Button(text="Rotate", background_color=(0.2, 0.6, 0.6, 1))
        self.b_live = ToggleButton(text="Live OFF", background_color=(0.3, 0.3, 0.3, 1))
        self.b_cap = Button(text="Capture", background_color=(0, 0.7, 0.4, 1))

        self.b_flip.bind(on_release=lambda *_: self.view.flip())
        self.b_rot.bind(on_release=lambda *_: self.view.rotate())
        self.b_live.bind(on_release=self.toggle_live)
        self.b_cap.bind(on_release=self.capture)

        for b in (self.b_flip, self.b_rot, self.b_live, self.b_cap):
            bar.add_widget(b)
        
        self.add_widget(bar)
        self.an = None
        Clock.schedule_once(self.load_models, 0.5)
        Clock.schedule_interval(self.live_tick, 1.0)

    def load_models(self, dt):
        try:
            self.an = Analyzer()
            count = len(self.an.known_database)
            self.label.text = f"Ready! Loaded {count} profile(s).\nTap Capture or Live ON."
        except Exception as ex:
            self.label.text = f"Model error: {ex}"

    def toggle_live(self, *_):
        on = self.b_live.state == "down"
        self.b_live.text = "Live ON" if on else "Live OFF"
        self.b_live.background_color = (0.8, 0.3, 0.1, 1) if on else (0.3, 0.3, 0.3, 1)
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
            return text
        except Exception as ex:
            self.label.text = f"Error: {ex}"
            return ""

    def send_email_async(self, img, analysis_text):
        """Sends an email with analysis details and captured image in a separate thread."""
        def background_send():
            try:
                msg = MIMEMultipart()
                msg['From'] = SENDER_EMAIL
                msg['To'] = ADMIN_EMAIL
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg['Subject'] = f"🚨 Selfie AI Alert: Capture Log - {timestamp}"

                body = f"""
                Hello Admin,
                
                A new capture/login event occurred at {timestamp}.
                
                Analysis Details:
                {analysis_text}
                
                Please find the captured photo attached.
                
                Regards,
                Selfie AI Pro System
                """
                msg.attach(MIMEText(body, 'plain'))

                # Encode image as attachment
                success, encoded_img = cv2.imencode('.jpg', img)
                if success:
                    image_data = encoded_img.tobytes()
                    attachment = MIMEImage(image_data, name="capture.jpg")
                    msg.attach(attachment)

                # Connect to SMTP server and send email securely via SSL
                with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                    server.login(SENDER_EMAIL, SENDER_PASSWORD)
                    server.sendmail(SENDER_EMAIL, ADMIN_EMAIL, msg.as_string())
                
                # Update UI on success
                Clock.schedule_once(lambda dt: setattr(self.label, 'text', analysis_text + "\n[✅ Email Sent Successfully!]"))
            except Exception as e:
                err_msg = str(e)
                print(f"Failed to send email: {err_msg}")
                # Update UI with error details if it fails
                Clock.schedule_once(lambda dt: setattr(self.label, 'text', analysis_text + f"\n[❌ Email Failed: {err_msg[:35]}]"))

        threading.Thread(target=background_send, daemon=True).start()

    def capture(self, *_):
        v = self.view
        if v.frozen:
            v.frozen, v.overlay = False, []
            self.b_cap.text = "Capture"
            self.label.text = "Ready."
            return
        if self.an is None or v.frame is None:
            self.label.text = "Models loading..."
            return
        
        analysis_text = self.run(v.frame)
        v.frozen = True
        v.show()
        self.b_cap.text = "Retake"
        
        # Trigger email notification to admin upon successful capture
        self.label.text = analysis_text + "\n[Sending Email...]"
        self.send_email_async(v.frame, analysis_text)


class SelfieApp(App):
    def build(self):
        self.title = "Selfie AI Pro"
        if platform == "android":
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.CAMERA, Permission.INTERNET])
        return Root()


if __name__ == "__main__":
    SelfieApp().run()
