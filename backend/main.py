import os
import gc
import asyncio

# Limitar los hilos de CPU para reducir sobrecarga
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from io import BytesIO

torch.set_num_threads(1)

app = FastAPI()

# Modelo preentrenado ligero
model = YOLO("yolo11n.pt")

# Evitar que se procesen varias fotos simultáneamente
prediction_lock = asyncio.Lock()

MAX_IMAGE_BYTES = 8 * 1024 * 1024


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


def ejecutar_inferencia(imagen):
    """
    Detecta personas y sillas.
    Devuelve coordenadas y confianzas, no los objetos
    completos de resultados de YOLO.
    """

    with torch.inference_mode():
        results = model.predict(
            source=imagen,
            conf=0.20,
            imgsz=416,
            max_det=20,
            classes=[0, 56],
            device="cpu",
            verbose=False
        )

        detecciones = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                clase = int(box.cls[0])
                confianza = float(box.conf[0])
                coordenadas = box.xyxy[0].tolist()

                detecciones.append({
                    "clase": clase,
                    "confianza": confianza,
                    "box": coordenadas
                })

        # Al salir de esta función se liberan los resultados
        # completos de YOLO y se conservan solo los datos simples.
        return detecciones


def compatibilidad_persona_silla(person_box, chair_box):
    """
    Calcula si una persona podría estar ocupando una silla.
    Devuelve una distancia: cuanto menor, mejor.
    Devuelve None cuando no hay coincidencia suficiente.
    """

    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = chair_box

    ancho_persona = max(px2 - px1, 1)
    alto_persona = max(py2 - py1, 1)

    ancho_silla = max(cx2 - cx1, 1)
    alto_silla = max(cy2 - cy1, 1)

    # Punto aproximado en la zona inferior de la persona.
    punto_x = (px1 + px2) / 2
    punto_y = py1 + alto_persona * 0.80

    # Ampliar la silla porque puede estar parcialmente oculta.
    margen_x = ancho_silla * 0.35
    margen_y = alto_silla * 0.40

    silla_x1 = cx1 - margen_x
    silla_y1 = cy1 - margen_y
    silla_x2 = cx2 + margen_x
    silla_y2 = cy2 + margen_y

    punto_dentro = (
        silla_x1 <= punto_x <= silla_x2
        and silla_y1 <= punto_y <= silla_y2
    )

    # Comprobar también la intersección de la silla
    # con la parte inferior de la caja de la persona.
    zona_persona_y1 = py1 + alto_persona * 0.55

    inter_x1 = max(px1, silla_x1)
    inter_y1 = max(zona_persona_y1, silla_y1)
    inter_x2 = min(px2, silla_x2)
    inter_y2 = min(py2, silla_y2)

    inter_ancho = max(0, inter_x2 - inter_x1)
    inter_alto = max(0, inter_y2 - inter_y1)

    area_interseccion = inter_ancho * inter_alto

    area_zona_persona = max(
        ancho_persona * (py2 - zona_persona_y1),
        1
    )

    porcentaje_solapamiento = (
        area_interseccion / area_zona_persona
    )

    if not punto_dentro and porcentaje_solapamiento < 0.12:
        return None

    # Priorizar la silla más próxima al punto de la persona.
    centro_silla_x = (cx1 + cx2) / 2
    centro_silla_y = (cy1 + cy2) / 2

    distancia_x = (
        punto_x - centro_silla_x
    ) / max(ancho_silla * 0.85, 1)

    distancia_y = (
        punto_y - centro_silla_y
    ) / max(alto_silla * 0.90, 1)

    return (distancia_x ** 2 + distancia_y ** 2) ** 0.5


def contar_sillas_ocupadas(personas, sillas):
    """
    Asigna cada persona como máximo a una silla
    y cada silla como máximo a una persona.
    """

    coincidencias = []

    for indice_persona, persona in enumerate(personas):
        for indice_silla, silla in enumerate(sillas):
            distancia = compatibilidad_persona_silla(
                persona["box"],
                silla["box"]
            )

            if distancia is not None:
                coincidencias.append((
                    distancia,
                    indice_persona,
                    indice_silla
                ))

    # Procesar primero las coincidencias más cercanas.
    coincidencias.sort(key=lambda item: item[0])

    personas_asignadas = set()
    sillas_asignadas = set()

    for _, indice_persona, indice_silla in coincidencias:
        if indice_persona in personas_asignadas:
            continue

        if indice_silla in sillas_asignadas:
            continue

        personas_asignadas.add(indice_persona)
        sillas_asignadas.add(indice_silla)

    return len(sillas_asignadas)


def analizar_imagen(imagen):
    """
    Detecta personas y sillas y prepara únicamente
    los números que necesita el sistema.
    """

    detecciones = ejecutar_inferencia(imagen)

    personas = []
    sillas = []

    for deteccion in detecciones:
        clase = deteccion["clase"]
        confianza = deteccion["confianza"]

        # Persona: exigir confianza mínima de 0.30.
        if clase == 0 and confianza >= 0.30:
            personas.append({
                "box": deteccion["box"],
                "confidence": confianza
            })

        # Silla: permitir detecciones desde 0.20.
        elif clase == 56 and confianza >= 0.20:
            sillas.append({
                "box": deteccion["box"],
                "confidence": confianza
            })

    # Contar ocupantes directamente, no usando las sillas.
    total_personas = len(personas)

    # Las sillas se detectan dinámicamente, sin fijar un total.
    total_sillas = len(sillas)

    sillas_ocupadas = contar_sillas_ocupadas(personas, sillas)

    sillas_libres = max(
        total_sillas - sillas_ocupadas,
        0
    )

    estado = "vacio" if total_personas == 0 else "ocupado"

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": sillas_ocupadas,
        "libres": sillas_libres,
        "estado": estado
    }

    # Liberar los datos intermedios antes de responder.
    del detecciones, personas, sillas
    gc.collect()

    return respuesta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    imagen = None
    image_bytes = b""

    try:
        # Leer una imagen de tamaño limitado.
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="La imagen está vacía"
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="La imagen supera el límite de 8 MB"
            )

        # No guardar la imagen en un archivo temporal.
        with Image.open(BytesIO(image_bytes)) as original:
            imagen = original.convert("RGB")

        # Reducir fotos demasiado grandes.
        imagen.thumbnail((640, 640))

        # Una sola inferencia al mismo tiempo.
        async with prediction_lock:
            respuesta = await asyncio.to_thread(
                analizar_imagen,
                imagen
            )

        return respuesta

    except HTTPException:
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido no es una imagen válida"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )

    finally:
        if imagen is not None:
            imagen.close()

        image_bytes = b""
        await file.close()
        gc.collect()