import os
import gc
import asyncio

# Reducir el uso de hilos de CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch

from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

torch.set_num_threads(1)

app = FastAPI()

# Modelo ligero: YOLO detectará solamente personas
model = YOLO("yolo11n.pt")

# Procesar una fotografía a la vez
prediction_lock = asyncio.Lock()

MAX_IMAGE_BYTES = 8 * 1024 * 1024

CONF_PERSONA = 0.30

# Rango HSV para los tonos verdes de las sillas
VERDE_MIN = np.array([25, 40, 35], dtype=np.uint8)
VERDE_MAX = np.array([95, 255, 255], dtype=np.uint8)


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando",
        "modelo": "YOLO11n + OpenCV",
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": "YOLO11n + OpenCV",
    }


def detectar_personas(imagen):
    """
    Ejecuta una sola inferencia de YOLO, exclusivamente
    para detectar personas.
    """

    personas = []

    with torch.inference_mode():
        resultados = model.predict(
            source=imagen,
            conf=CONF_PERSONA,
            imgsz=416,
            max_det=20,
            classes=[0],
            device="cpu",
            verbose=False,
        )

        for resultado in resultados:
            if resultado.boxes is None:
                continue

            for caja in resultado.boxes:
                x1, y1, x2, y2 = caja.xyxy[0].tolist()

                personas.append({
                    "id": len(personas) + 1,
                    "confianza": float(caja.conf[0]),
                    "box": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                })

        del resultados

    return personas


def detectar_sillas_verdes(imagen):
    """
    Detecta regiones verdes mediante HSV y componentes
    conectados. No utiliza un número fijo de sillas.

    Devuelve las regiones candidatas y un diagnóstico
    de los componentes verdes encontrados.
    """

    imagen_rgb = np.asarray(imagen)
    alto, ancho = imagen_rgb.shape[:2]
    area_imagen = max(ancho * alto, 1)

    imagen_hsv = cv2.cvtColor(
        imagen_rgb,
        cv2.COLOR_RGB2HSV,
    )

    mascara = cv2.inRange(
        imagen_hsv,
        VERDE_MIN,
        VERDE_MAX,
    )

    # Limpiar pequeñas manchas de color y cerrar huecos pequeños
    kernel_apertura = np.ones((3, 3), dtype=np.uint8)
    kernel_cierre = np.ones((5, 5), dtype=np.uint8)

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel_apertura,
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel_cierre,
    )

    cantidad, etiquetas, estadisticas, centroides = (
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8,
        )
    )

    # Umbral mínimo relativo al tamaño de la imagen.
    # No establece cuántas sillas tiene que haber.
    area_minima = max(
        120,
        int(area_imagen * 0.0008),
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
                "motivo": "componente_demasiado_pequeno",
            })
            continue

        # Contar cuántos píxeles del rectángulo son realmente verdes.
        region = etiquetas[y:y + h, x:x + w]
        proporcion_verde = float(
            np.count_nonzero(region == indice) / max(w * h, 1)
        )

        silla = {
            "id": len(sillas) + 1,
            "box": [
                float(x),
                float(y),
                float(x + w),
                float(y + h),
            ],
            "area_verde": area,
            "proporcion_verde": round(proporcion_verde, 3),
            "centro": [
                round(float(centro_x), 1),
                round(float(centro_y), 1),
            ],
        }

        sillas.append(silla)

    diagnostico = {
        "componentes_verdes_totales": cantidad - 1,
        "area_minima_pixeles": area_minima,
        "componentes_descartados": componentes_descartados,
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
    """
    Estima si una zona verde pertenece a una silla
    ocupada por una persona.

    Se expande la caja de la silla para tener en cuenta
    que el cuerpo puede ocultar parte de su superficie.
    """

    sx1, sy1, sx2, sy2 = silla_box
    px1, py1, px2, py2 = persona_box

    ancho_silla = max(sx2 - sx1, 1)
    alto_silla = max(sy2 - sy1, 1)

    margen_x = ancho_silla * 0.20
    margen_y = alto_silla * 0.15

    silla_expandida = [
        sx1 - margen_x,
        sy1 - margen_y,
        sx2 + margen_x,
        sy2 + margen_y,
    ]

    # La parte inferior del cuerpo también se considera
    # para asociar la persona con su silla.
    alto_persona = max(py2 - py1, 1)

    zona_inferior_persona = [
        px1,
        py1 + alto_persona * 0.35,
        px2,
        py2,
    ]

    interseccion_total = interseccion_cajas(
        persona_box,
        silla_expandida,
    )

    interseccion_inferior = interseccion_cajas(
        zona_inferior_persona,
        silla_expandida,
    )

    area_persona = max(area_caja(persona_box), 1)
    area_silla = max(area_caja(silla_expandida), 1)

    solapamiento_total = interseccion_total / min(
        area_persona,
        area_silla,
    )

    solapamiento_inferior = interseccion_inferior / area_silla

    return max(
        solapamiento_total,
        solapamiento_inferior,
    )


def asociar_personas_sillas(personas, sillas):
    """
    Asigna como máximo una silla a cada persona y
    como máximo una persona a cada silla.
    """

    candidatos = []

    for persona in personas:
        for silla in sillas:
            puntuacion = relacion_persona_silla(
                persona["box"],
                silla["box"],
            )

            if puntuacion >= 0.05:
                candidatos.append({
                    "persona": persona["id"],
                    "silla": silla["id"],
                    "puntuacion": round(puntuacion, 3),
                })

    candidatos.sort(
        key=lambda item: item["puntuacion"],
        reverse=True,
    )

    personas_asignadas = set()
    sillas_asignadas = set()
    asignaciones = []

    for candidato in candidatos:
        persona_id = candidato["persona"]
        silla_id = candidato["silla"]

        if (
            persona_id in personas_asignadas
            or silla_id in sillas_asignadas
        ):
            candidato["asignada"] = False
            continue

        candidato["asignada"] = True

        personas_asignadas.add(persona_id)
        sillas_asignadas.add(silla_id)

        asignaciones.append(dict(candidato))

    return (
        len(sillas_asignadas),
        asignaciones,
        candidatos,
        [
            p["id"]
            for p in personas
            if p["id"] not in personas_asignadas
        ],
        [
            s["id"]
            for s in sillas
            if s["id"] not in sillas_asignadas
        ],
    )


def analizar_imagen(imagen):
    # Una inferencia de YOLO, exclusivamente para personas.
    personas = detectar_personas(imagen)

    # OpenCV busca las sillas utilizando su color verde.
    sillas, diagnostico_verde = detectar_sillas_verdes(imagen)

    (
        ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_silla,
        sillas_sin_persona,
    ) = asociar_personas_sillas(personas, sillas)

    total_personas = len(personas)
    total_sillas = len(sillas)
    libres = max(total_sillas - ocupadas, 0)

    estado = (
        "vacio"
        if total_personas == 0
        else "ocupado"
    )

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": ocupadas,
        "libres": libres,
        "estado": estado,
        "diagnostico": {
            "metodo_personas": "YOLO11n",
            "metodo_sillas": "OpenCV HSV color verde",
            "inferencias_yolo": 1,
            "personas_detectadas": [
                {
                    "id": p["id"],
                    "confianza": round(p["confianza"], 3),
                    "caja": [
                        round(v, 1) for v in p["box"]
                    ],
                }
                for p in personas
            ],
            "sillas_detectadas": [
                {
                    "id": s["id"],
                    "caja": [
                        round(v, 1) for v in s["box"]
                    ],
                    "area_verde": s["area_verde"],
                    "proporcion_verde": s["proporcion_verde"],
                }
                for s in sillas
            ],
            "deteccion_color": diagnostico_verde,
            "asignaciones": asignaciones,
            "coincidencias_evaluadas": coincidencias,
            "personas_sin_silla": personas_sin_silla,
            "sillas_sin_persona": sillas_sin_persona,
        },
    }

    del personas, sillas
    gc.collect()

    return respuesta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    imagen = None
    image_bytes = b""

    try:
        image_bytes = await file.read(MAX_IMAGE_BYTES + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="La imagen está vacía",
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="La imagen supera el límite de 8 MB",
            )

        # No se guarda ningún archivo temporal.
        with Image.open(BytesIO(image_bytes)) as original:
            imagen = original.convert("RGB")

        imagen.thumbnail((640, 640))

        async with prediction_lock:
            respuesta = await asyncio.to_thread(
                analizar_imagen,
                imagen,
            )

        return respuesta

    except HTTPException:
        raise

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="El archivo recibido no es una imagen válida",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}",
        )

    finally:
        if imagen is not None:
            imagen.close()

        image_bytes = b""
        await file.close()
        gc.collect()