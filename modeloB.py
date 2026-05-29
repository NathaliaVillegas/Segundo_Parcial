import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import RPi.GPIO as GPIO

BUTTON_PIN = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

interpreter = tf.lite.Interpreter(model_path="model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo cargad")

classes = []
with open("labels.txt", "r", encoding="utf-8") as f:
    for linea in f:
        linea = linea.strip()
        if linea:
            partes = linea.split(" ", 1)
            if len(partes) > 1:
                classes.append(partes[1])
            else:
                classes.append(partes[0])

print("\nClases cargadas:")
for i, c in enumerate(classes):
    print(i, c)

conteo = {}
for clase in classes:
    conteo[clase] = 0

uclase = "Nada"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("No camara")
    exit()

inicio = time.time()
tiempo_limite = 60
tiempo_pausado = 0
inicio_pausa = 0
emergencia = False
ultimo_estado_boton = 1

print("\ninicio 60 segundos")

while True:
    estado_boton = GPIO.input(BUTTON_PIN)
    if estado_boton == 0 and ultimo_estado_boton == 1:
        time.sleep(0.05)
        if GPIO.input(BUTTON_PIN) == 0:
            if not emergencia:
                emergencia = True
                inicio_pausa = time.time()
                print("Pausado")
            else:
                emergencia = False
                tiempo_pausado += time.time() - inicio_pausa
                print("Reanudado")
    ultimo_estado_boton = estado_boton

    if not emergencia:
        tiempo_actual = time.time() - inicio - tiempo_pausado
        if tiempo_actual >= tiempo_limite:
            break
    else:
        tiempo_actual = inicio_pausa - inicio - tiempo_pausado

    ret, frame = cap.read()
    if not ret:
        break
    vista = frame.copy()

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == 27:
        break

    if emergencia:
        cv2.rectangle(vista, (0, 0), (vista.shape[1], vista.shape[0]), (0, 0, 255), 10)
        cv2.putText(vista, "PARADA DE EMERGENCIA", (90, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        cv2.imshow("Detector de Botellas", vista)
        continue

    imagen = cv2.resize(frame, (224, 224))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen = np.asarray(imagen, dtype=np.float32)
    imagen = (imagen / 127.5) - 1
    imagen = np.expand_dims(imagen, axis=0)

    interpreter.set_tensor(input_details[0]["index"], imagen)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]["index"])

    indice = np.argmax(pred[0])
    confianza = pred[0][indice]
    clase = classes[indice]

    if confianza < 0.80:
        clase = "Nada"

    if clase != uclase:
        if clase != "Nada":
            conteo[clase] += 1
            print(f"Detect: {clase}")
        uclase = clase

    texto = f"{clase} {confianza * 100:.1f}%"
    cv2.putText(vista, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    tiempo_restante = int(tiempo_limite - tiempo_actual)
    cv2.putText(vista, f"Time: {tiempo_restante}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Detector de Botellas", vista)

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()

print("RESULTADOS FINALES")
for k, v in conteo.items():
    print(f"{k}: {v}")

with open("resultado_conteo.txt", "w") as archivo:
    archivo.write("RESULTADO DE BOTELLAS\n")
    for k, v in conteo.items():
        archivo.write(f"{k}: {v}\n")

print("\nTXT: conteo.txt")

top3 = sorted(conteo.items(), key=lambda x: x[1], reverse=True)[:3]
nombres = [x[0] for x in top3]
cantidades = [x[1] for x in top3]

plt.figure(figsize=(8, 5))
barras = plt.bar(nombres, cantidades)
plt.title("3 botellas mas detectadas")
plt.xlabel("botella")
plt.ylabel("cantidad")

for barra in barras:
    altura = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, altura, str(int(altura)), ha="center")

plt.savefig("top3.png")
print("PNG: top3.png")
plt.show()
