import os
import gc
import asyncio

# Limitar los hilos de CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
from io import BytesIO

torch.set_num_threads(1)

app = FastAPI()

# Modelo ligero
model = YOLO("yolo11n.pt")

# Procesar una sola imagen a la vez
prediction_lock = asyncio.Lock()

MAX_IMAGE_BYTES = 8 * 1024 * 1024

CONF_PERSONA = 0.30
CONF_SILLA = 0.20
CONF_SILLA_OCULTA = 0.08

UMBRAL_CONTENCION = 0.85
UMBRAL_IOU_DUPLICADO = 0.15
UMBRAL_ASOCIACION = 0.15
UMBRAL_SILLA_OCULTA = 0.20


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
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def area_interseccion(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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


def contencion_cajas(box_a, box_b):
    area_menor = min(
        area_caja(box_a),
        area_caja(box_b)
    )

    if area_menor <= 0:
        return 0.0

    return area_interseccion(box_a, box_b) / area_menor


def ejecutar_inferencia(imagen):
    """
    Una sola llamada a YOLO para la fotografía completa.
    No crea recortes ni archivos temporales.
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

    gc.collect()
    return detecciones


def filtrar_detecciones(
    detecciones,
    clase_objetivo,
    confianza_minima,
    usar_iou=False
):
    """
    Filtra por clase y confianza y elimina cajas
    casi completamente contenidas en otras detecciones.
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
    descartadas = []

    for candidata in candidatas:
        mejor = None

        for aceptada in seleccionadas:
            contencion = contencion_cajas(
                candidata["box"],
                aceptada["box"]
            )

            iou = calcular_iou(
                candidata["box"],
                aceptada["box"]
            )

            es_duplicado = (
                contencion >= UMBRAL_CONTENCION
                or (
                    usar_iou
                    and iou >= 0.40
                )
            )

            if es_duplicado:
                if (
                    mejor is None
                    or contencion > mejor["contencion"]
                ):
                    mejor = {
                        "referencia": aceptada,
                        "contencion": contencion,
                        "iou": iou
                    }

        if mejor is not None:
            descartadas.append({
                "id_modelo": candidata["id_modelo"],
                "clase": (
                    "persona"
                    if clase_objetivo == 0
                    else "silla"
                ),
                "confianza": round(
                    candidata["confianza"], 3
                ),
                "duplicado_de": (
                    mejor["referencia"]["id_modelo"]
                ),
                "contencion": round(
                    mejor["contencion"], 3
                ),
                "iou": round(mejor["iou"], 3)
            })
        else:
            seleccionadas.append(candidata)

    return seleccionadas, descartadas


def buscar_duplicado_silla(candidata, sillas):
    """
    Revisa si una candidata de baja confianza
    ya está representada por otra caja.
    """

    mejor = None

    for silla in sillas:
        contencion = contencion_cajas(
            candidata["box"],
            silla["box"]
        )

        iou = calcular_iou(
            candidata["box"],
            silla["box"]
        )

        if (
            contencion >= UMBRAL_CONTENCION
            or iou >= UMBRAL_IOU_DUPLICADO
        ):
            if (
                mejor is None
                or contencion > mejor["contencion"]
            ):
                mejor = {
                    "referencia": silla,
                    "contencion": contencion,
                    "iou": iou
                }

    return mejor


def asociar_personas_sillas(personas, sillas):
    """
    Una persona y una silla pueden formar como máximo
    una pareja en el resultado final.
    """

    coincidencias = []

    for persona in personas:
        for silla in sillas:
            solapamiento = contencion_cajas(
                persona["box"],
                silla["box"]
            )

            if solapamiento >= UMBRAL_ASOCIACION:
                coincidencias.append({
                    "persona": persona["id"],
                    "silla": silla["id"],
                    "solapamiento": round(
                        solapamiento, 3
                    )
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

        asignada = (
            persona_id not in personas_asignadas
            and silla_id not in sillas_asignadas
        )

        coincidencia["asignada"] = asignada

        if asignada:
            personas_asignadas.add(persona_id)
            sillas_asignadas.add(silla_id)
            asignaciones.append(dict(coincidencia))

    personas_sin_silla = [
        persona["id"]
        for persona in personas
        if persona["id"] not in personas_asignadas
    ]

    sillas_sin_persona = [
        silla["id"]
        for silla in sillas
        if silla["id"] not in sillas_asignadas
    ]

    return (
        len(sillas_asignadas),
        asignaciones,
        coincidencias,
        personas_sin_silla,
        sillas_sin_persona
    )


def analizar_imagen(imagen):
    """
    Analiza la fotografía una sola vez con YOLO.
    Admite candidatas de silla de baja confianza cuando
    hay una persona sin silla y la coincidencia espacial
    resulta suficientemente clara.
    """

    detecciones = ejecutar_inferencia(imagen)

    personas_seleccionadas, duplicados_personas = (
        filtrar_detecciones(
            detecciones,
            clase_objetivo=0,
            confianza_minima=CONF_PERSONA,
            usar_iou=True
        )
    )

    sillas_seleccionadas, duplicados_sillas = (
        filtrar_detecciones(
            detecciones,
            clase_objetivo=56,
            confianza_minima=CONF_SILLA,
            usar_iou=False
        )
    )

    personas = [
        {
            "id": indice + 1,
            "id_modelo": d["id_modelo"],
            "box": d["box"],
            "confidence": d["confianza"]
        }
        for indice, d in enumerate(personas_seleccionadas)
    ]

    sillas = [
        {
            "id": indice + 1,
            "id_modelo": d["id_modelo"],
            "box": d["box"],
            "confidence": d["confianza"],
            "origen": "modelo",
            "respaldada_por_persona": None
        }
        for indice, d in enumerate(sillas_seleccionadas)
    ]

    (
        _,
        asignaciones_iniciales,
        _,
        personas_pendientes,
        _
    ) = asociar_personas_sillas(personas, sillas)

    candidatas_bajas = [
        dict(d)
        for d in detecciones
        if (
            d["clase"] == 56
            and CONF_SILLA_OCULTA <= d["confianza"] < CONF_SILLA
        )
    ]

    candidatas_bajas.sort(
        key=lambda d: d["confianza"],
        reverse=True
    )

    cajas_procesadas = list(sillas)
    personas_pendientes = set(personas_pendientes)

    diagnostico_candidatas = []
    duplicados_bajos = []

    for candidata in candidatas_bajas:
        duplicado = buscar_duplicado_silla(
            candidata,
            cajas_procesadas
        )

        if duplicado is not None:
            referencia = duplicado["referencia"]

            duplicados_bajos.append({
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

            diagnostico_candidatas.append({
                "id_modelo": candidata["id_modelo"],
                "confianza": round(
                    candidata["confianza"], 3
                ),
                "resultado": "duplicada",
                "persona_relacionada": None
            })

            # Registrar también las candidatas descartadas
            # para no reintroducir cajas prácticamente idénticas.
            cajas_procesadas.append(candidata)
            continue

        persona_soporte = None
        mejor_solapamiento = 0.0

        for persona in personas:
            if persona["id"] not in personas_pendientes:
                continue

            solapamiento = contencion_cajas(
                candidata["box"],
                persona["box"]
            )

            if (
                solapamiento >= UMBRAL_SILLA_OCULTA
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

            diagnostico_candidatas.append({
                "id_modelo": candidata["id_modelo"],
                "confianza": round(
                    candidata["confianza"], 3
                ),
                "resultado": "promovida_por_solapamiento",
                "persona_relacionada": persona_soporte,
                "solapamiento": round(
                    mejor_solapamiento, 3
                )
            })
        else:
            diagnostico_candidatas.append({
                "id_modelo": candidata["id_modelo"],
                "confianza": round(
                    candidata["confianza"], 3
                ),
                "resultado": "sin_persona_sin_silla_compatible",
                "persona_relacionada": None
            })

        cajas_procesadas.append(candidata)

    (
        ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_silla,
        sillas_sin_persona
    ) = asociar_personas_sillas(personas, sillas)

    total_personas = len(personas)
    total_sillas = len(sillas)
    libres = max(total_sillas - ocupadas, 0)

    estado = "vacio" if total_personas == 0 else "ocupado"

    ids_incluidos = {
        p["id_modelo"] for p in personas
    } | {
        s["id_modelo"] for s in sillas
    }

    duplicados = duplicados_personas + duplicados_sillas + duplicados_bajos

    duplicado_por_id = {
        d["id_modelo"]: d["duplicado_de"]
        for d in duplicados
    }

    diagnostico_detecciones = []

    for d in detecciones:
        clase = d["clase"]
        umbral = CONF_PERSONA if clase == 0 else CONF_SILLA

        diagnostico_detecciones.append({
            "id_modelo": d["id_modelo"],
            "clase": "persona" if clase == 0 else "silla",
            "confianza": round(d["confianza"], 3),
            "umbral_conteo_normal": umbral,
            "supera_umbral": d["confianza"] >= umbral,
            "incluida_en_conteo": (
                d["id_modelo"] in ids_incluidos
            ),
            "duplicado_de": duplicado_por_id.get(
                d["id_modelo"]
            ),
            "caja": [
                round(v, 1) for v in d["box"]
            ]
        })

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": ocupadas,
        "libres": libres,
        "estado": estado,
        "diagnostico": {
            "inferencias_yolo": 1,
            "detecciones_modelo": diagnostico_detecciones,
            "personas_detectadas": [
                {
                    "id": p["id"],
                    "id_modelo": p["id_modelo"],
                    "confianza": round(p["confidence"], 3),
                    "caja": [
                        round(v, 1) for v in p["box"]
                    ]
                }
                for p in personas
            ],
            "sillas_detectadas": [
                {
                    "id": s["id"],
                    "id_modelo": s["id_modelo"],
                    "confianza": round(s["confidence"], 3),
                    "origen": s["origen"],
                    "respaldada_por_persona": (
                        s["respaldada_por_persona"]
                    ),
                    "caja": [
                        round(v, 1) for v in s["box"]
                    ]
                }
                for s in sillas
            ],
            "duplicados_descartados": duplicados,
            "candidatas_bajas_confianza": diagnostico_candidatas,
            "asignaciones_iniciales": asignaciones_iniciales,
            "asignaciones": asignaciones,
            "coincidencias_evaluadas": coincidencias,
            "personas_sin_silla": personas_sin_silla,
            "sillas_sin_persona": sillas_sin_persona
        }
    }

    del detecciones
    del personas
    del sillas
    del personas_seleccionadas
    del sillas_seleccionadas

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

        # No se crea temp.jpg ni se almacenan las fotos en disco.
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