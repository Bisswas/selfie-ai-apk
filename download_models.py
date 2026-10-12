"""Run once on your PC:  python download_models.py
Downloads the 9 model files into ./models. URLs are from memory and
unverified - if one fails, search the file name on GitHub and place it manually."""
import os
import urllib.request

OPENCV = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
FACE_W = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/"
AG = "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/"
MSSD = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/"
ONNX = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/"

FILES = {
    "face_deploy.prototxt": OPENCV + "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": FACE_W + "res10_300x300_ssd_iter_140000.caffemodel",
    "age_deploy.prototxt": AG + "age_deploy.prototxt",
    "age_net.caffemodel": AG + "age_net.caffemodel",
    "gender_deploy.prototxt": AG + "gender_deploy.prototxt",
    "gender_net.caffemodel": AG + "gender_net.caffemodel",
    "emotion-ferplus-8.onnx": ONNX + "emotion-ferplus-8.onnx",
    "MobileNetSSD_deploy.prototxt": MSSD + "deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": MSSD + "mobilenet_iter_73000.caffemodel",
}

os.makedirs("models", exist_ok=True)
for name, url in FILES.items():
    path = os.path.join("models", name)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        print("skip  ", name)
        continue
    try:
        urllib.request.urlretrieve(url, path)
        print("ok    ", name, os.path.getsize(path) // 1024, "KB")
    except Exception as e:
        print("FAILED", name, "-", e)
