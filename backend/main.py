import os
import gc
import asyncio

# Reducir la sobrecarga de CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from io import BytesIO

torch.set_num_threads(1)

app = FastAPI()

# Modelo ligero preentrenado
model = YOLO("yolo11n.pt")

# Evitar inferencias simultáneas
prediction_lock = asyncio.Lock()

# Tamaño máximo permitido: 8 MB
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
    Detecta personas y sillas y devuelve únicamente
    las coordenadas, clases y confianzas.
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

        del results
        return detecciones


def compatibilidad_persona_silla(person_box, chair_box):
    """
    Estima la compatibilidad espacial entre una persona
    y una silla. Una distancia menor significa una
    coincidencia geométrica más cercana.
    """

    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = chair_box

    ancho_persona = max(px2 - px1, 1)
    alto_persona = max(py2 - py1, 1)

    ancho_silla = max(cx2 - cx1, 1)
    alto_silla = max(cy2 - cy1, 1)

    # Punto aproximado de la parte inferior de la persona
    punto_x = (px1 + px2) / 2
    punto_y = py1 + alto_persona * 0.80

    # Ampliar la caja de la silla para considerar oclusiones
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

    # Comprobar intersección con la zona inferior de la persona
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
    Busca parejas posibles entre personas y sillas.
    Una persona no puede ocupar dos sillas y una silla
    no puede asignarse a dos personas.

    Devuelve el conteo y los datos de diagnóstico.
    """

    coincidencias = []

    for indice_persona, persona in enumerate(personas):
        for indice_silla, silla in enumerate(sillas):

            distancia = compatibilidad_persona_silla(
                persona["box"],
                silla["box"]
            )

            if distancia is not None:
                coincidencias.append({
                    "persona": indice_persona + 1,
                    "silla": indice_silla + 1,
                    "distancia": round(distancia, 3)
                })

    # Priorizar las coincidencias geométricamente más cercanas
    coincidencias.sort(key=lambda item: item["distancia"])

    personas_asignadas = set()
    sillas_asignadas = set()
    asignaciones = []

    for coincidencia in coincidencias:
        persona_id = coincidencia["persona"]
        silla_id = coincidencia["silla"]

        if (
            persona_id not in personas_asignadas
            and silla_id not in sillas_asignadas
        ):
            coincidencia["asignada"] = True

            personas_asignadas.add(persona_id)
            sillas_asignadas.add(silla_id)

            asignaciones.append(dict(coincidencia))
        else:
            coincidencia["asignada"] = False

    personas_con_candidata = {
        item["persona"] for item in coincidencias
    }

    sillas_con_candidata = {
        item["silla"] for item in coincidencias
    }

    personas_sin_coincidencia = [
        i
        for i in range(1, len(personas) + 1)
        if i not in personas_con_candidata
    ]

    sillas_sin_coincidencia = [
        i
        for i in range(1, len(sillas) + 1)
        if i not in sillas_con_candidata
    ]

    return (
        len(sillas_asignadas),
        asignaciones,
        coincidencias,
        personas_sin_coincidencia,
        sillas_sin_coincidencia
    )


def analizar_imagen(imagen):
    """
    Detecta personas y sillas, calcula la ocupación
    y devuelve información adicional para diagnosticar
    los errores de detección y asociación.
    """

    detecciones = ejecutar_inferencia(imagen)

    personas = []
    sillas = []

    # Aplicar los umbrales finales de confianza
    for deteccion in detecciones:
        clase = deteccion["clase"]
        confianza = deteccion["confianza"]

        if clase == 0 and confianza >= 0.30:
            personas.append({
                "box": deteccion["box"],
                "confidence": confianza
            })

        elif clase == 56 and confianza >= 0.20:
            sillas.append({
                "box": deteccion["box"],
                "confidence": confianza
            })

    total_personas = len(personas)
    total_sillas = len(sillas)

    (
        sillas_ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_coincidencia,
        sillas_sin_coincidencia
    ) = contar_sillas_ocupadas(personas, sillas)

    sillas_libres = max(
        total_sillas - sillas_ocupadas,
        0
    )

    estado = "vacio" if total_personas == 0 else "ocupado"

    # Registrar todas las detecciones de YOLO antes
    # de aplicar los umbrales adicionales del conteo.
    detecciones_diagnostico = []

    for deteccion in detecciones:
        clase = deteccion["clase"]
        confianza = deteccion["confianza"]

        detecciones_diagnostico.append({
            "clase": "persona" if clase == 0 else "silla",
            "confianza": round(confianza, 3),
            "incluida_en_conteo": (
                confianza >= 0.30
                if clase == 0
                else confianza >= 0.20
            ),
            "caja": [
                round(valor, 1)
                for valor in deteccion["box"]
            ]
        })

    personas_diagnostico = [
        {
            "id": indice + 1,
            "confianza": round(persona["confidence"], 3),
            "caja": [
                round(valor, 1)
                for valor in persona["box"]
            ]
        }
        for indice, persona in enumerate(personas)
    ]

    sillas_diagnostico = [
        {
            "id": indice + 1,
            "confianza": round(silla["confidence"], 3),
            "caja": [
                round(valor, 1)
                for valor in silla["box"]
            ]
        }
        for indice, silla in enumerate(sillas)
    ]

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": sillas_ocupadas,
        "libres": sillas_libres,
        "estado": estado,
        "diagnostico": {
            "detecciones_modelo": detecciones_diagnostico,
            "personas_detectadas": personas_diagnostico,
            "sillas_detectadas": sillas_diagnostico,
            "asignaciones": asignaciones,
            "coincidencias_evaluadas": coincidencias,
            "personas_sin_coincidencia": personas_sin_coincidencia,
            "sillas_sin_coincidencia": sillas_sin_coincidencia
        }
    }

    # Liberar datos intermedios antes de responder
    del detecciones
    del personas
    del sillas
    gc.collect()

    return respuesta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    imagen = None
    image_bytes = b""

    try:
        # Limitar el tamaño de la imagen recibida
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

        # Procesar la imagen directamente en memoria
        with Image.open(BytesIO(image_bytes)) as original:
            imagen = original.convert("RGB")

        # Reducir imágenes excesivamente grandes
        imagen.thumbnail((640, 640))

        # Solo una inferencia simultánea
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