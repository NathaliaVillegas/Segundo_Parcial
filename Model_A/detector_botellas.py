import cv2
import numpy as np
import tensorflow as tf
import serial
import time
from threading import Thread

#
MODEL_PATH = "bebidas_model.keras"
URL = "http://192.168.26.2:8080/video"
class_names = ["coca", "fanta", "pepsi", "salvietti"]

try:
    ser = serial.Serial('/dev/serial0', 9600, timeout=0.1)
    time.sleep(2)
except:
    ser = None

#Hilitos
class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if grabbed: self.frame = frame
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True
        self.stream.release()

# Inicialización
vs = VideoStream(URL).start()
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# Variables de control:
last_seen_time = time.time()

while True:
    frame = vs.read()
    if frame is None: continue

    H, W = frame.shape[:2]
    display = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.addWeighted(gray, 0.55, np.full_like(gray, 135), 0.45, 0)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    found_label = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 33000 or area > (H * W * 0.70): continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        roi = frame[max(0, y-20):min(H, y+ch+20), max(0, x-20):min(W, x+cw+20)]
        if roi.size == 0: continue
        
        pred = model.predict(preprocess(roi), verbose=0)[0]
        idx = np.argmax(pred)
        if pred[idx] > 0.70:
            found_label = class_names[idx]
            # Mostrar detección en pantalla (tu código original)
            cv2.putText(display, f"{found_label} {pred[idx]:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            break 

    if found_label:
        last_seen_time = time.time()
        
        char_to_send = ""
        if found_label == "coca": char_to_send = "c"
        elif found_label == "salvietti": char_to_send = "s"
        elif found_label == "pepsi": char_to_send = "p"
        elif found_label == "fanta": char_to_send = "f"
        
        if ser and char_to_send != "":
            ser.write((char_to_send + "\n").encode())
            ser.flush()

    else:
        if (time.time() - last_seen_time) > 3.0:
            if ser:
                ser.write("m\n".encode())
                ser.flush()

    if not found_label:
        cv2.putText(display, "No botellas", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camara", cv2.resize(display, (720, 1280)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vs.stop()
cv2.destroyAllWindows()
if ser: ser.close()