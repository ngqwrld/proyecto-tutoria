
import os
import gc
import asyncio
import json
import logging
import time
import uuid

# Reducir los hilos utilizados por PyTorch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch

from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

torch.set_num_threads(1)

app = FastAPI()

# Medir la carga del modelo al iniciar
inicio_carga_modelo = time.perf_counter()
model = YOLO("yolo11n.pt")
CARGA_MODELO_MS = round(
    (time.perf_counter() - inicio_carga_modelo) * 1000, 2
)

print(
    json.dumps({
        "evento": "modelo_cargado",
        "modelo": "YOLO11n",
        "carga_modelo_ms": CARGA_MODELO_MS
    }),
    flush=True
)

# Una sola inferencia a la vez
prediction_lock = asyncio.Lock()

MAX_IMAGE_BYTES = 8 * 1024 * 1024
CONF_PERSONA = 0.30

# Tonos verdes en espacio HSV
VERDE_MIN = np.array([25, 40, 35], dtype=np.uint8)
VERDE_MAX = np.array([95, 255, 255], dtype=np.uint8)


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando",
        "modelo": "YOLO11n + OpenCV",
        "medicion_tiempos": True
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": "YOLO11n + OpenCV"
    }


def detectar_personas(imagen):
    """Detecta personas y mide por separado la inferencia YOLO."""

    inicio = time.perf_counter()
    personas = []

    with torch.inference_mode():
        resultados = model.predict(
            source=imagen,
            conf=CONF_PERSONA,
            imgsz=416,
            max_det=20,
            classes=[0],
            device="cpu",
            verbose=False
        )

        for resultado in resultados:
            if resultado.boxes is None:
                continue

            for caja in resultado.boxes:
                x1, y1, x2, y2 = caja.xyxy[0].tolist()

                personas.append({
                    "id": len(personas) + 1,
                    "confianza": float(caja.conf[0]),
                    "box": [float(x1), float(y1),
                            float(x2), float(y2)]
                })

        del resultados

    tiempo_ms = round(
        (time.perf_counter() - inicio) * 1000, 2
    )

    return personas, tiempo_ms


def detectar_sillas_verdes(imagen):
    """
    Busca componentes de color verde con OpenCV.
    Este método se mantiene sin cambios para medir
    el rendimiento real de la versión actual.
    """

    imagen_rgb = np.asarray(imagen)
    alto, ancho = imagen_rgb.shape[:2]
    area_imagen = max(ancho * alto, 1)

    imagen_hsv = cv2.cvtColor(
        imagen_rgb, cv2.COLOR_RGB2HSV
    )

    mascara = cv2.inRange(
        imagen_hsv, VERDE_MIN, VERDE_MAX
    )

    kernel_apertura = np.ones((3, 3), dtype=np.uint8)
    kernel_cierre = np.ones((5, 5), dtype=np.uint8)

    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_OPEN, kernel_apertura
    )

    mascara = cv2.morphologyEx(
        mascara, cv2.MORPH_CLOSE, kernel_cierre
    )

    cantidad, etiquetas, estadisticas, centroides = (
        cv2.connectedComponentsWithStats(
            mascara, connectivity=8
        )
    )

    area_minima = max(
        120, int(area_imagen * 0.0008)
    )

    sillas = []
    componentes_descartados = []

    for indice in range(1, cantidad):
        x = int(estadisticas[indice, cv2.CC_STAT_LEFT])
        y = int(estadisticas[indice, cv2.CC_STAT_TOP])
        w = int(estadisticas[indice, cv2.CC_STAT_WIDTH])
        h = int(estadisticas[indice, cv2.CC_STAT_HEIGHT])
        area = int(estadisticas[indice, cv2.CC_STAT_AREA])

        centro_x, centro_y = centroides[indice]

        if area < area_minima or w < 8 or h < 8:
            componentes_descartados.append({
                "area_pixeles": area,
                "caja": [x, y, x + w, y + h],
                "motivo": "componente_demasiado_pequeno"
            })
            continue

        region = etiquetas[y:y + h, x:x + w]
        proporcion_verde = float(
            np.count_nonzero(region == indice) / max(w * h, 1)
        )

        sillas.append({
            "id": len(sillas) + 1,
            "box": [
                float(x), float(y),
                float(x + w), float(y + h)
            ],
            "area_verde": area,
            "proporcion_verde": round(proporcion_verde, 3),
            "centro": [
                round(float(centro_x), 1),
                round(float(centro_y), 1)
            ]
        })

    diagnostico = {
        "componentes_verdes_totales": cantidad - 1,
        "area_minima_pixeles": area_minima,
        "componentes_descartados": componentes_descartados
    }

    del imagen_hsv, mascara, etiquetas
    gc.collect()

    return sillas, diagnostico


def interseccion_cajas(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def area_caja(box):
    x1, y1, x2, y2 = box
    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def relacion_persona_silla(persona_box, silla_box):
    px1, py1, px2, py2 = persona_box
    sx1, sy1, sx2, sy2 = silla_box

    ancho_silla = max(sx2 - sx1, 1)
    alto_silla = max(sy2 - sy1, 1)

    silla_expandida = [
        sx1 - ancho_silla * 0.20,
        sy1 - alto_silla * 0.15,
        sx2 + ancho_silla * 0.20,
        sy2 + alto_silla * 0.15
    ]

    alto_persona = max(py2 - py1, 1)

    zona_inferior_persona = [
        px1, py1 + alto_persona * 0.35, px2, py2
    ]

    interseccion_total = interseccion_cajas(
        persona_box, silla_expandida
    )

    interseccion_inferior = interseccion_cajas(
        zona_inferior_persona, silla_expandida
    )

    area_persona = max(area_caja(persona_box), 1)
    area_silla = max(area_caja(silla_expandida), 1)

    total = interseccion_total / min(
        area_persona, area_silla
    )

    inferior = interseccion_inferior / area_silla

    return max(total, inferior)


def asociar_personas_sillas(personas, sillas):
    candidatos = []

    for persona in personas:
        for silla in sillas:
            puntuacion = relacion_persona_silla(
                persona["box"], silla["box"]
            )

            if puntuacion >= 0.05:
                candidatos.append({
                    "persona": persona["id"],
                    "silla": silla["id"],
                    "puntuacion": round(puntuacion, 3)
                })

    candidatos.sort(
        key=lambda item: item["puntuacion"],
        reverse=True
    )

    personas_asignadas = set()
    sillas_asignadas = set()
    asignaciones = []

    for candidato in candidatos:
        persona_id = candidato["persona"]
        silla_id = candidato["silla"]

        asignada = (
            persona_id not in personas_asignadas
            and silla_id not in sillas_asignadas
        )

        candidato["asignada"] = asignada

        if asignada:
            personas_asignadas.add(persona_id)
            sillas_asignadas.add(silla_id)
            asignaciones.append(dict(candidato))

    personas_sin_silla = [
        p["id"] for p in personas
        if p["id"] not in personas_asignadas
    ]

    sillas_sin_persona = [
        s["id"] for s in sillas
        if s["id"] not in sillas_asignadas
    ]

    return (
        len(sillas_asignadas),
        asignaciones,
        candidatos,
        personas_sin_silla,
        sillas_sin_persona
    )


def analizar_imagen(imagen):
    """Mide por separado YOLO, OpenCV, asociación y análisis total."""

    inicio_analisis = time.perf_counter()

    personas, yolo_ms = detectar_personas(imagen)

    inicio_opencv = time.perf_counter()
    sillas, diagnostico_verde = detectar_sillas_verdes(imagen)
    opencv_ms = round(
        (time.perf_counter() - inicio_opencv) * 1000, 2
    )

    inicio_asociacion = time.perf_counter()

    (
        ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_silla,
        sillas_sin_persona
    ) = asociar_personas_sillas(personas, sillas)

    asociacion_ms = round(
        (time.perf_counter() - inicio_asociacion) * 1000, 2
    )

    total_personas = len(personas)
    total_sillas = len(sillas)
    libres = max(total_sillas - ocupadas, 0)

    estado = "vacio" if total_personas == 0 else "ocupado"

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": ocupadas,
        "libres": libres,
        "estado": estado,
        "diagnostico": {
            "metodo_personas": "YOLO11n",
            "metodo_sillas": "OpenCV HSV verde",
            "personas_detectadas": [
                {
                    "id": p["id"],
                    "confianza": round(p["confianza"], 3),
                    "caja": [round(v, 1) for v in p["box"]]
                }
                for p in personas
            ],
            "sillas_detectadas": [
                {
                    "id": s["id"],
                    "caja": [round(v, 1) for v in s["box"]],
                    "area_verde": s["area_verde"],
                    "proporcion_verde": s["proporcion_verde"]
                }
                for s in sillas
            ],
            "deteccion_color": diagnostico_verde,
            "asignaciones": asignaciones,
            "coincidencias_evaluadas": coincidencias,
            "personas_sin_silla": personas_sin_silla,
            "sillas_sin_persona": sillas_sin_persona
        },
        "tiempos_ms": {
            "yolo_inferencia_ms": yolo_ms,
            "opencv_sillas_ms": opencv_ms,
            "asociacion_ms": asociacion_ms
        }
    }

    respuesta["tiempos_ms"]["analisis_total_ms"] = round(
        (time.perf_counter() - inicio_analisis) * 1000, 2
    )

    del personas, sillas
    gc.collect()

    return respuesta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    inicio_api = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]

    imagen = None
    image_bytes = b""

    lectura_ms = 0
    decodificacion_ms = 0
    espera_lock_ms = 0

    try:
        inicio_lectura = time.perf_counter()
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
        lectura_ms = round(
            (time.perf_counter() - inicio_lectura) * 1000, 2
        )

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

        inicio_decodificacion = time.perf_counter()

        with Image.open(BytesIO(image_bytes)) as original:
            imagen = original.convert("RGB")

        imagen.thumbnail((640, 640))

        decodificacion_ms = round(
            (time.perf_counter() - inicio_decodificacion) * 1000, 2
        )

        inicio_espera = time.perf_counter()

        async with prediction_lock:
            espera_lock_ms = round(
                (time.perf_counter() - inicio_espera) * 1000, 2
            )

            respuesta = await asyncio.to_thread(
                analizar_imagen, imagen
            )

        tiempos = respuesta["tiempos_ms"]

        tiempos["lectura_bytes_ms"] = lectura_ms
        tiempos["decodificacion_imagen_ms"] = decodificacion_ms
        tiempos["espera_turno_ms"] = espera_lock_ms
        tiempos["api_total_ms"] = round(
            (time.perf_counter() - inicio_api) * 1000, 2
        )

        registro = {
            "evento": "predict_completado",
            "request_id": request_id,
            "personas": respuesta["personas"],
            "sillas": respuesta["sillas"],
            "tiempos_ms": tiempos
        }

        logger.info(json.dumps(registro))

        return respuesta

    except HTTPException as error:
        logger.warning(json.dumps({
            "evento": "predict_http_error",
            "request_id": request_id,
            "status_code": error.status_code,
            "api_total_ms": round(
                (time.perf_counter() - inicio_api) * 1000, 2
            )
        }))
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido no es una imagen válida"
        )

    except Exception:
        logger.exception(json.dumps({
            "evento": "predict_error",
            "request_id": request_id
        }))
        raise HTTPException(
            status_code=500,
            detail="Error procesando la imagen. Revisa los logs del servidor."
        )

    finally:
        if imagen is not None:
            imagen.close()

        image_bytes = b""
        await file.close()
        gc.collect()
