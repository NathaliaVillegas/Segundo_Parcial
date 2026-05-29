import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import RPi.GPIO as GPIO
from threading import Thread

class VideoStream:
    def __init__(self, src):
        self.stream = cv2.VideoCapture(src)
        self.frame = None
        self.stopped = False
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if ret: self.frame = frame
    def read(self):
        return self.frame
    def stop(self):
        self.stopped = True
        self.stream.release()

URL = "http://192.168.26.2:8080/video"
BUTTON_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

MODEL_PATH = "bebidas_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["coca", "fanta", "pepsi", "salvietti"]
conteo = {c: 0 for c in class_names}
uclase = "Nada"
vs = VideoStream(URL).start()
time.sleep(2)

inicio = time.time()
tiempo_limite = 60
tiempo_pausado = 0
inicio_pausa = 0
emergencia = False
ultimo_estado_boton = 1

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

try:
    while True:
        frame = vs.read()
        if frame is None: continue

        estado_boton = GPIO.input(BUTTON_PIN)
        if estado_boton == 0 and ultimo_estado_boton == 1:
            time.sleep(0.05)
            if GPIO.input(BUTTON_PIN) == 0:
                if not emergencia:
                    emergencia = True
                    inicio_pausa = time.time()
                else:
                    emergencia = False
                    tiempo_pausado += time.time() - inicio_pausa
        ultimo_estado_boton = estado_boton

        if not emergencia:
            tiempo_actual = time.time() - inicio - tiempo_pausado
            if tiempo_actual >= tiempo_limite: break
        else:
            tiempo_actual = inicio_pausa - inicio - tiempo_pausado

        vista = frame.copy()

        if emergencia:
            cv2.rectangle(vista, (0, 0), (vista.shape[1], vista.shape[0]), (0, 0, 255), 10)
            cv2.putText(vista, "PARADA DE EMERGENCIA", (90, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            cv2.imshow("Detector de Botellas", vista)
            cv2.waitKey(1)
            continue

        H, W = frame.shape[:2]
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
                cv2.putText(vista, f"{found_label} {pred[idx]:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break

        if found_label:
            if found_label != uclase and found_label != "Nada":
                conteo[found_label] += 1
            uclase = found_label
        else:
            uclase = "Nada"
            cv2.putText(vista, "No botellas", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Detector de Botellas", vista)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    vs.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()

    with open("resultado_conteo.txt", "w") as f:
        for k, v in conteo.items():
            f.write(f"{k}: {v}\n")

    top3 = sorted(conteo.items(), key=lambda x: x[1], reverse=True)[:3]
    plt.bar([x[0] for x in top3], [x[1] for x in top3])
    plt.title("Top 3 botellas")
    plt.show()