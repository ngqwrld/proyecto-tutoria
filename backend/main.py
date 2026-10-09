import os
import gc
import asyncio

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from io import BytesIO

torch.set_num_threads(1)

app = FastAPI()

model = YOLO("yolo11n.pt")
prediction_lock = asyncio.Lock()

MAX_IMAGE_BYTES = 8 * 1024 * 1024

CONF_PERSONA = 0.30
CONF_SILLA = 0.20
CONF_SILLA_OCULTA = 0.08

UMBRAL_CONTENCION_DUPLICADO = 0.85
UMBRAL_IOU_DUPLICADO = 0.15
UMBRAL_SOLAPAMIENTO_ASOCIACION = 0.15
UMBRAL_SOLAPAMIENTO_SILLA_OCULTA = 0.20


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


def area_caja(box):
    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def area_interseccion(box_a, box_b):
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


def proporcion_solapamiento_menor(box_a, box_b):
    """
    Proporción de la caja más pequeña cubierta
    por la intersección de ambas cajas.
    """

    area_menor = min(
        area_caja(box_a),
        area_caja(box_b)
    )

    if area_menor <= 0:
        return 0.0

    return area_interseccion(box_a, box_b) / area_menor


def calcular_iou(box_a, box_b):
    interseccion = area_interseccion(box_a, box_b)

    union = (
        area_caja(box_a)
        + area_caja(box_b)
        - interseccion
    )

    if union <= 0:
        return 0.0

    return interseccion / union


def buscar_duplicado(candidata, existentes, umbral_iou=None):
    """
    Busca una caja casi contenida en otra o, cuando se indica,
    una caja con IoU suficiente para considerarla duplicada.
    """

    mejor = None

    for existente in existentes:
        contencion = proporcion_solapamiento_menor(
            candidata["box"],
            existente["box"]
        )

        iou = calcular_iou(
            candidata["box"],
            existente["box"]
        )

        es_duplicado = (
            contencion >= UMBRAL_CONTENCION_DUPLICADO
            or (
                umbral_iou is not None
                and iou >= umbral_iou
            )
        )

        if es_duplicado:
            if mejor is None or contencion > mejor["contencion"]:
                mejor = {
                    "referencia": existente,
                    "contencion": contencion,
                    "iou": iou
                }

    return mejor


def ejecutar_inferencia(imagen):
    """
    Ejecuta YOLO y devuelve detecciones de personas y sillas,
    incluidas las de menor confianza para el diagnóstico.
    """

    with torch.inference_mode():
        results = model.predict(
            source=imagen,
            conf=0.05,
            imgsz=416,
            max_det=30,
            classes=[0, 56],
            device="cpu",
            verbose=False
        )

        detecciones = []
        siguiente_id = 1

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                clase = int(box.cls[0])
                confianza = float(box.conf[0])
                coordenadas = box.xyxy[0].tolist()

                detecciones.append({
                    "id_modelo": siguiente_id,
                    "clase": clase,
                    "confianza": confianza,
                    "box": coordenadas
                })

                siguiente_id += 1

        del results
        return detecciones


def filtrar_detecciones(
    detecciones,
    clase_objetivo,
    confianza_minima
):
    """
    Filtra por clase y confianza y elimina duplicados
    casi completamente contenidos en detecciones mejores.
    """

    candidatas = [
        dict(d)
        for d in detecciones
        if (
            d["clase"] == clase_objetivo
            and d["confianza"] >= confianza_minima
        )
    ]

    candidatas.sort(
        key=lambda d: d["confianza"],
        reverse=True
    )

    seleccionadas = []
    duplicados = []

    nombre_clase = (
        "persona" if clase_objetivo == 0 else "silla"
    )

    for candidata in candidatas:
        duplicado = buscar_duplicado(
            candidata,
            seleccionadas
        )

        if duplicado:
            referencia = duplicado["referencia"]

            duplicados.append({
                "id_modelo": candidata["id_modelo"],
                "clase": nombre_clase,
                "confianza": round(
                    candidata["confianza"], 3
                ),
                "duplicado_de": referencia["id_modelo"],
                "contencion": round(
                    duplicado["contencion"], 3
                ),
                "iou": round(duplicado["iou"], 3)
            })
        else:
            seleccionadas.append(candidata)

    return seleccionadas, duplicados


def asociar_personas_sillas(personas, sillas):
    """
    Asocia personas y sillas usando el solapamiento entre
    sus cajas, una estrategia más adecuada para sillas
    parcialmente ocultas por las personas.

    Cada persona y cada silla solo pueden asignarse una vez.
    """

    coincidencias = []

    for persona in personas:
        for silla in sillas:
            solapamiento = proporcion_solapamiento_menor(
                persona["box"],
                silla["box"]
            )

            if solapamiento >= UMBRAL_SOLAPAMIENTO_ASOCIACION:
                coincidencias.append({
                    "persona": persona["id"],
                    "silla": silla["id"],
                    "solapamiento": round(solapamiento, 3)
                })

    coincidencias.sort(
        key=lambda item: item["solapamiento"],
        reverse=True
    )

    personas_asignadas = set()
    sillas_asignadas = set()
    asignaciones = []

    for coincidencia in coincidencias:
        persona_id = coincidencia["persona"]
        silla_id = coincidencia["silla"]

        if (
            persona_id in personas_asignadas
            or silla_id in sillas_asignadas
        ):
            coincidencia["asignada"] = False
            continue

        coincidencia["asignada"] = True

        personas_asignadas.add(persona_id)
        sillas_asignadas.add(silla_id)

        asignaciones.append(dict(coincidencia))

    personas_sin_coincidencia = [
        persona["id"]
        for persona in personas
        if persona["id"] not in personas_asignadas
    ]

    sillas_sin_coincidencia = [
        silla["id"]
        for silla in sillas
        if silla["id"] not in sillas_asignadas
    ]

    return (
        len(sillas_asignadas),
        asignaciones,
        coincidencias,
        personas_sin_coincidencia,
        sillas_sin_coincidencia
    )


def agregar_sillas_ocultas(
    detecciones,
    personas,
    sillas,
    personas_sin_coincidencia
):
    """
    Considera sillas de baja confianza únicamente si:
    - no duplican una silla ya conocida;
    - se solapan suficientemente con una persona sin silla;
    - no se ha añadido ya otra candidata para esa persona.

    No establece un número fijo de sillas.
    """

    candidatas = [
        dict(d)
        for d in detecciones
        if (
            d["clase"] == 56
            and CONF_SILLA_OCULTA <= d["confianza"] < CONF_SILLA
        )
    ]

    candidatas.sort(
        key=lambda d: d["confianza"],
        reverse=True
    )

    # Estas cajas permiten suprimir detecciones repetidas,
    # incluso cuando la detección anterior también era débil.
    cajas_procesadas = list(sillas)

    personas_pendientes = set(personas_sin_coincidencia)

    diagnostico_candidatas = []
    duplicados = []

    for candidata in candidatas:
        motivo = None
        persona_soporte = None
        mejor_solapamiento = 0.0

        duplicado = buscar_duplicado(
            candidata,
            cajas_procesadas,
            umbral_iou=UMBRAL_IOU_DUPLICADO
        )

        if duplicado:
            referencia = duplicado["referencia"]

            motivo = "duplicada"

            duplicados.append({
                "id_modelo": candidata["id_modelo"],
                "clase": "silla",
                "confianza": round(
                    candidata["confianza"], 3
                ),
                "duplicado_de": referencia["id_modelo"],
                "contencion": round(
                    duplicado["contencion"], 3
                ),
                "iou": round(duplicado["iou"], 3)
            })

            cajas_procesadas.append(candidata)

        else:
            # Buscar una persona sin silla que respalde
            # espacialmente esta posible silla oculta.
            for persona in personas:
                if persona["id"] not in personas_pendientes:
                    continue

                solapamiento = proporcion_solapamiento_menor(
                    candidata["box"],
                    persona["box"]
                )

                if (
                    solapamiento >= UMBRAL_SOLAPAMIENTO_SILLA_OCULTA
                    and solapamiento > mejor_solapamiento
                ):
                    mejor_solapamiento = solapamiento
                    persona_soporte = persona["id"]

            if persona_soporte is not None:
                nueva_silla = {
                    "id": len(sillas) + 1,
                    "id_modelo": candidata["id_modelo"],
                    "box": candidata["box"],
                    "confidence": candidata["confianza"],
                    "origen": "candidata_oculta",
                    "respaldada_por_persona": persona_soporte
                }

                sillas.append(nueva_silla)
                personas_pendientes.remove(persona_soporte)

                motivo = "promovida_por_solapamiento"
                cajas_procesadas.append(candidata)

            else:
                motivo = "sin_persona_sin_silla_compatible"
                cajas_procesadas.append(candidata)

        diagnostico_candidatas.append({
            "id_modelo": candidata["id_modelo"],
            "confianza": round(candidata["confianza"], 3),
            "resultado": motivo,
            "persona_relacionada": persona_soporte,
            "solapamiento": round(mejor_solapamiento, 3)
        })

    return diagnostico_candidatas, duplicados


def analizar_imagen(imagen):
    """
    Calcula personas, sillas, ocupación y diagnóstico.
    """

    detecciones = ejecutar_inferencia(imagen)

    personas_seleccionadas, duplicados_personas = (
        filtrar_detecciones(
            detecciones,
            clase_objetivo=0,
            confianza_minima=CONF_PERSONA
        )
    )

    sillas_seleccionadas, duplicados_sillas = (
        filtrar_detecciones(
            detecciones,
            clase_objetivo=56,
            confianza_minima=CONF_SILLA
        )
    )

    personas = [
        {
            "id": indice + 1,
            "id_modelo": deteccion["id_modelo"],
            "box": deteccion["box"],
            "confidence": deteccion["confianza"]
        }
        for indice, deteccion in enumerate(personas_seleccionadas)
    ]

    sillas = [
        {
            "id": indice + 1,
            "id_modelo": deteccion["id_modelo"],
            "box": deteccion["box"],
            "confidence": deteccion["confianza"],
            "origen": "modelo",
            "respaldada_por_persona": None
        }
        for indice, deteccion in enumerate(sillas_seleccionadas)
    ]

    # Primero asociar las sillas de confianza normal.
    (
        _,
        asignaciones_iniciales,
        _,
        personas_sin_coincidencia,
        _
    ) = asociar_personas_sillas(personas, sillas)

    # Revisar candidatas débiles que podrían representar
    # sillas parcialmente ocultas.
    candidatas_bajas, duplicados_bajos = agregar_sillas_ocultas(
        detecciones,
        personas,
        sillas,
        personas_sin_coincidencia
    )

    # Recalcular las asociaciones con todas las sillas válidas.
    (
        ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_coincidencia,
        sillas_sin_coincidencia
    ) = asociar_personas_sillas(personas, sillas)

    total_personas = len(personas)
    total_sillas = len(sillas)

    libres = max(total_sillas - ocupadas, 0)

    estado = (
        "vacio" if total_personas == 0
        else "ocupado"
    )

    duplicados = (
        duplicados_personas
        + duplicados_sillas
        + duplicados_bajos
    )

    duplicado_por_id = {
        item["id_modelo"]: item["duplicado_de"]
        for item in duplicados
    }

    ids_incluidos = {
        persona["id_modelo"]
        for persona in personas
    } | {
        silla["id_modelo"]
        for silla in sillas
    }

    diagnostico_detecciones = []

    for deteccion in detecciones:
        clase = deteccion["clase"]

        umbral = (
            CONF_PERSONA if clase == 0
            else CONF_SILLA
        )

        diagnostico_detecciones.append({
            "id_modelo": deteccion["id_modelo"],
            "clase": "persona" if clase == 0 else "silla",
            "confianza": round(deteccion["confianza"], 3),
            "umbral_conteo_normal": umbral,
            "supera_umbral": deteccion["confianza"] >= umbral,
            "incluida_en_conteo": (
                deteccion["id_modelo"] in ids_incluidos
            ),
            "duplicado_de": duplicado_por_id.get(
                deteccion["id_modelo"]
            ),
            "caja": [
                round(valor, 1)
                for valor in deteccion["box"]
            ]
        })

    personas_diagnostico = [
        {
            "id": persona["id"],
            "id_modelo": persona["id_modelo"],
            "confianza": round(persona["confidence"], 3),
            "caja": [
                round(valor, 1)
                for valor in persona["box"]
            ]
        }
        for persona in personas
    ]

    sillas_diagnostico = [
        {
            "id": silla["id"],
            "id_modelo": silla["id_modelo"],
            "confianza": round(silla["confidence"], 3),
            "origen": silla["origen"],
            "respaldada_por_persona": (
                silla["respaldada_por_persona"]
            ),
            "caja": [
                round(valor, 1)
                for valor in silla["box"]
            ]
        }
        for silla in sillas
    ]

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": ocupadas,
        "libres": libres,
        "estado": estado,
        "diagnostico": {
            "detecciones_modelo": diagnostico_detecciones,
            "personas_detectadas": personas_diagnostico,
            "sillas_detectadas": sillas_diagnostico,
            "duplicados_descartados": duplicados,
            "candidatas_bajas_confianza": candidatas_bajas,
            "asignaciones_iniciales": asignaciones_iniciales,
            "asignaciones": asignaciones,
            "coincidencias_evaluadas": coincidencias,
            "personas_sin_coincidencia": personas_sin_coincidencia,
            "sillas_sin_coincidencia": sillas_sin_coincidencia
        }
    }

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

        # Procesamiento directo en memoria, sin temp.jpg
        with Image.open(BytesIO(image_bytes)) as original:
            imagen = original.convert("RGB")

        imagen.thumbnail((640, 640))

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