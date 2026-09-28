from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import os

app = FastAPI()

model = YOLO("yolo11n.pt")


@app.get("/")
def inicio():
    return {"mensaje": "Servidor de tutoria funcionando"}


@app.get("/health")
def health():
    return {"estado": "ok"}


def punto_en_silla(person_box, chair_box):
    """
    Comprueba si la parte inferior de una persona
    está dentro de la zona de una silla.

    person_box = [x1, y1, x2, y2]
    chair_box = [x1, y1, x2, y2]
    """

    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = chair_box

    # Punto usado: centro de la parte inferior
    punto_x = (px1 + px2) / 2
    punto_y = py2

    # Ampliamos ligeramente la zona de la silla
    ancho = cx2 - cx1
    alto = cy2 - cy1

    margen_x = ancho * 0.25
    margen_y = alto * 0.50

    cx1_ampliado = cx1 - margen_x
    cx2_ampliado = cx2 + margen_x
    cy1_ampliado = cy1 - margen_y
    cy2_ampliado = cy2 + margen_y

    return (
        cx1_ampliado <= punto_x <= cx2_ampliado
        and
        cy1_ampliado <= punto_y <= cy2_ampliado
    )


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

    # -----------------------------------
    # DETERMINAR SILLAS OCUPADAS
    # -----------------------------------

    occupied_chairs = 0

    for chair in chairs:

        chair_box = chair["box"]

        silla_ocupada = False

        for person in people:

            person_box = person["box"]

            if punto_en_silla(person_box, chair_box):
                silla_ocupada = True
                break

        if silla_ocupada:
            occupied_chairs += 1

    # -----------------------------------
    # RESULTADOS
    # -----------------------------------

    total_chairs = len(chairs)

    free_chairs = total_chairs - occupied_chairs

    # IMPORTANTE:
    # No usamos el número bruto de personas detectadas
    # por YOLO porque puede detectar personas que no
    # están ocupando una silla.
    people_count = occupied_chairs

    if os.path.exists("temp.jpg"):
        os.remove("temp.jpg")

    return {
        "personas": people_count,
        "sillas": total_chairs,
        "ocupadas": occupied_chairs,
        "libres": free_chairs
    }
