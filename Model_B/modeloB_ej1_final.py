import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

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

print("\ninicio 60 segundos")


while time.time() - inicio < 60:
    ret, frame = cap.read()
    if not ret:
        break
    vista = frame.copy()

    imagen = cv2.resize(frame,(224, 224))
    imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
    imagen = np.asarray(imagen,dtype=np.float32)
    imagen = (imagen / 127.5) - 1
    imagen = np.expand_dims(imagen,axis=0)

    interpreter.set_tensor(
        input_details[0]["index"],
        imagen)

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

    texto = (
        f"{clase} "
        f"{confianza * 100:.1f}%")

    cv2.putText(vista,texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,   1,  (0, 255, 0), 2)
    tiempo_restante = int(60 - (time.time() - inicio))
    cv2.putText(vista,f"Time: {tiempo_restante}s", (10, 80),cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 0),2)
    cv2.imshow("Detector de Botellas", vista )

    tecla = cv2.waitKey(1)

    if tecla == 27:
        break


cap.release()
cv2.destroyAllWindows()

print("RESULTADOS FINALES")

for k, v in conteo.items():
    print(f"{k}: {v}")

with open("resultado_conteo.txt", "w" ) as archivo:

    archivo.write("RESULTADO DE BOTELLAS\n")

    for k, v in conteo.items():

        archivo.write(f"{k}: {v}\n")

print("\nTXT: conteo.txt")

top3 = sorted(conteo.items(),key=lambda x: x[1],reverse=True)[:3]

nombres = [x[0] for x in top3]
cantidades = [x[1] for x in top3]


plt.figure(figsize=(8, 5))

barras = plt.bar(nombres,cantidades)

plt.title("3 botellas mas detectadas")

plt.xlabel("botella")
plt.ylabel("cantidad")

for barra in barras:

    altura = barra.get_height()

    plt.text(barra.get_x() + barra.get_width()/2, altura, str(int(altura)), ha="center")

plt.savefig("top3.png")

print("PNG: top3.png")

plt.show()