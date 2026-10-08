import os
import gc
import asyncio

# Reducir consumo de CPU/RAM de PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import torch

torch.set_num_threads(1)

app = FastAPI()

# Modelo ligero
model = YOLO("yolo11n.pt")

# Evita dos análisis simultáneos y un aumento innecesario de RAM
prediction_lock = asyncio.Lock()


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando",
        "modelo": "YOLO11n"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": "YOLO11n"
    }


def punto_en_silla(person_box, chair_box):
    """
    Comprueba si la parte inferior de una persona
    coincide con la zona de una silla.
    """

    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = chair_box

    punto_x = (px1 + px2) / 2
    punto_y = py2

    ancho = cx2 - cx1
    alto = cy2 - cy1

    margen_x = ancho * 0.25
    margen_y = alto * 0.50

    return (
        cx1 - margen_x <= punto_x <= cx2 + margen_x
        and
        cy1 - margen_y <= punto_y <= cy2 + margen_y
    )


def analizar_imagen(imagen):
    """
    Ejecuta YOLO y devuelve solamente los datos necesarios.
    No conserva el objeto completo de resultados.
    """

    with torch.inference_mode():

        results = model.predict(
            source=imagen,
            conf=0.30,
            imgsz=320,
            max_det=20,
            device="cpu",
            verbose=False
        )

        people = []
        chairs = []

        for result in results:

            if result.boxes is None:
                continue

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

        # Asociar personas con sillas
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

        # IMPORTANTE:
        # Las personas se cuentan directamente.
        people_count = len(people)

        total_chairs = len(chairs)

        free_chairs = max(
            total_chairs - occupied_chairs,
            0
        )

        estado = (
            "vacio"
            if people_count == 0
            else "ocupado"
        )

        resultado = {
            "personas": people_count,
            "sillas": total_chairs,
            "ocupadas": occupied_chairs,
            "libres": free_chairs,
            "estado": estado
        }

        # Liberar resultados inmediatamente
        del results
        del people
        del chairs

        gc.collect()

        return resultado


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ninguna imagen"
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="La imagen está vacía"
        )

    try:

        # Abrir directamente desde memoria
        imagen = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # No conservar imágenes enormes
        imagen.thumbnail((640, 640))

        # Solo una predicción simultánea
        async with prediction_lock:

            resultado = await asyncio.to_thread(
                analizar_imagen,
                imagen
            )

        # Liberar imagen
        imagen.close()

        del image_bytes
        del imagen

        gc.collect()

        return resultado

    except Exception as e:

        gc.collect()

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )