from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import os

app = FastAPI()

model = YOLO("yolo11n.pt")


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok"
    }


def calcular_iou(box1, box2):
    """
    Calcula cuánto se superponen dos cajas.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    ancho = max(0, x2 - x1)
    alto = max(0, y2 - y1)

    interseccion = ancho * alto

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - interseccion

    if union == 0:
        return 0

    return interseccion / union


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    with open("temp.jpg", "wb") as f:
        f.write(image_bytes)

    results = model("temp.jpg")

    people = []
    chairs = []

    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            coordinates = box.xyxy[0].tolist()

            # Persona
            if class_id == 0:
                people.append({
                    "box": coordinates,
                    "confidence": confidence
                })

            # Silla
            elif class_id == 56:
                chairs.append({
                    "box": coordinates,
                    "confidence": confidence
                })

    # Determinar sillas ocupadas
    occupied_chairs = 0

    for chair in chairs:

        chair_box = chair["box"]
        silla_ocupada = False

        for person in people:

            person_box = person["box"]

            iou = calcular_iou(chair_box, person_box)

            # Si la persona se superpone suficientemente
            # con la silla, consideramos que está ocupada.
            if iou >= 0.10:
                silla_ocupada = True
                break

        if silla_ocupada:
            occupied_chairs += 1

    free_chairs = len(chairs) - occupied_chairs

    # Eliminar imagen temporal
    if os.path.exists("temp.jpg"):
        os.remove("temp.jpg")

    return {
        "personas": len(people),
        "sillas": len(chairs),
        "ocupadas": occupied_chairs,
        "libres": free_chairs
    }
