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
CONF_SILLA_RECORTE = 0.08

MAX_AREA_SILLA = 0.35
UMBRAL_CONTENCION = 0.82
UMBRAL_IOU_PERSONA = 0.45
UMBRAL_IOU_SILLA = 0.25
UMBRAL_ASOCIACION = 0.15


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
    return max(0, x2 - x1) * max(0, y2 - y1)


def area_interseccion(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0, x2 - x1) * max(0, y2 - y1)


def iou_cajas(box_a, box_b):
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


def ejecutar_yolo(
    imagen,
    clases,
    confianza,
    origen,
    contador_ids,
    offset_x=0,
    offset_y=0
):
    """
    Ejecuta YOLO sobre una imagen o recorte.
    Convierte las coordenadas de recortes a las
    coordenadas de la imagen completa.
    """

    detecciones = []

    with torch.inference_mode():
        results = model.predict(
            source=imagen,
            conf=confianza,
            imgsz=416,
            max_det=30,
            classes=clases,
            device="cpu",
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                clase = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                contador_ids[0] += 1

                detecciones.append({
                    "id_modelo": contador_ids[0],
                    "clase": clase,
                    "confianza": score,
                    "origen": origen,
                    "box": [
                        x1 + offset_x,
                        y1 + offset_y,
                        x2 + offset_x,
                        y2 + offset_y
                    ]
                })

        del results

    gc.collect()
    return detecciones


def eliminar_duplicados(
    candidatas,
    umbral_contencion=UMBRAL_CONTENCION,
    umbral_iou=0.40
):
    """
    Elimina cajas casi idénticas conservando primero
    las detecciones con mayor confianza.
    """

    ordenadas = sorted(
        candidatas,
        key=lambda item: item["confianza"],
        reverse=True
    )

    seleccionadas = []
    descartadas = []

    for candidata in ordenadas:
        duplicado_de = None
        mejor_contencion = 0.0
        mejor_iou = 0.0

        for aceptada in seleccionadas:
            contencion = contencion_cajas(
                candidata["box"],
                aceptada["box"]
            )

            iou = iou_cajas(
                candidata["box"],
                aceptada["box"]
            )

            if (
                contencion >= umbral_contencion
                or iou >= umbral_iou
            ):
                if (
                    duplicado_de is None
                    or contencion > mejor_contencion
                ):
                    duplicado_de = aceptada
                    mejor_contencion = contencion
                    mejor_iou = iou

        if duplicado_de is not None:
            descartadas.append({
                "id_modelo": candidata["id_modelo"],
                "clase": candidata["clase"],
                "confianza": round(
                    candidata["confianza"], 3
                ),
                "origen": candidata["origen"],
                "duplicado_de": duplicado_de["id_modelo"],
                "contencion": round(
                    mejor_contencion, 3
                ),
                "iou": round(mejor_iou, 3)
            })
        else:
            seleccionadas.append(candidata)

    return seleccionadas, descartadas


def detectar_sillas_en_recortes(imagen, contador_ids):
    """
    Divide la imagen en cuatro recortes superpuestos.
    Conserva las cajas en coordenadas de la imagen original.
    """

    ancho, alto = imagen.size

    ancho_recorte = max(1, int(ancho * 0.62))
    alto_recorte = max(1, int(alto * 0.62))

    posiciones_x = sorted({
        0,
        max(0, ancho - ancho_recorte)
    })

    posiciones_y = sorted({
        0,
        max(0, alto - alto_recorte)
    })

    detecciones = []
    recortes_info = []
    cajas_grandes = []

    for y0 in posiciones_y:
        for x0 in posiciones_x:
            x1 = min(x0 + ancho_recorte, ancho)
            y1 = min(y0 + alto_recorte, alto)

            nombre = f"recorte_x{x0}_y{y0}"

            recorte = imagen.crop((x0, y0, x1, y1))

            try:
                resultados = ejecutar_yolo(
                    imagen=recorte,
                    clases=[56],
                    confianza=0.05,
                    origen=nombre,
                    contador_ids=contador_ids,
                    offset_x=x0,
                    offset_y=y0
                )
            finally:
                recorte.close()

            validas = 0
            grandes = 0

            for deteccion in resultados:
                fraccion_area = (
                    area_caja(deteccion["box"])
                    / max(ancho * alto, 1)
                )

                deteccion["fraccion_area_imagen"] = round(
                    fraccion_area, 3
                )

                if fraccion_area > MAX_AREA_SILLA:
                    cajas_grandes.append({
                        "id_modelo": deteccion["id_modelo"],
                        "confianza": round(
                            deteccion["confianza"], 3
                        ),
                        "origen": nombre,
                        "fraccion_area_imagen": round(
                            fraccion_area, 3
                        ),
                        "motivo": "caja_demasiado_grande",
                        "box": deteccion["box"]
                    })
                    grandes += 1
                    continue

                if deteccion["confianza"] >= CONF_SILLA_RECORTE:
                    detecciones.append(deteccion)
                    validas += 1

            recortes_info.append({
                "origen": nombre,
                "coordenadas": [x0, y0, x1, y1],
                "detecciones_validas": validas,
                "cajas_grandes_descartadas": grandes
            })

            del resultados
            gc.collect()

    return detecciones, recortes_info, cajas_grandes


def asociar_personas_sillas(personas, sillas):
    """
    Asocia personas con sillas mediante la superposición
    de sus cajas. Cada persona y silla se asigna una vez.
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
    ancho, alto = imagen.size
    area_imagen = max(ancho * alto, 1)

    contador_ids = [0]

    # Primera pasada: imagen completa, personas y sillas.
    detecciones_completas = ejecutar_yolo(
        imagen=imagen,
        clases=[0, 56],
        confianza=0.05,
        origen="imagen_completa",
        contador_ids=contador_ids
    )

    personas_raw = [
        d for d in detecciones_completas
        if (
            d["clase"] == 0
            and d["confianza"] >= CONF_PERSONA
        )
    ]

    personas, duplicados_personas = eliminar_duplicados(
        personas_raw,
        umbral_contencion=0.85,
        umbral_iou=UMBRAL_IOU_PERSONA
    )

    # Las cajas de silla que abarcan gran parte de la imagen
    # se tratan como sospechosas, no como una silla confirmada.
    sillas_completas = []
    cajas_grandes_completas = []

    for deteccion in detecciones_completas:
        if deteccion["clase"] != 56:
            continue

        fraccion_area = (
            area_caja(deteccion["box"]) / area_imagen
        )

        if fraccion_area > MAX_AREA_SILLA:
            cajas_grandes_completas.append({
                "id_modelo": deteccion["id_modelo"],
                "confianza": round(
                    deteccion["confianza"], 3
                ),
                "origen": deteccion["origen"],
                "fraccion_area_imagen": round(
                    fraccion_area, 3
                ),
                "motivo": "caja_demasiado_grande",
                "box": deteccion["box"]
            })
            continue

        if deteccion["confianza"] >= CONF_SILLA:
            sillas_completas.append(deteccion)

    # Solo hacer análisis por recortes si la detección
    # de la imagen completa parece insuficiente.
    usar_recortes = (
        len(sillas_completas) < 2
        or len(cajas_grandes_completas) > 0
    )

    detecciones_recortes = []
    recortes_info = []
    cajas_grandes_recortes = []

    if usar_recortes:
        (
            detecciones_recortes,
            recortes_info,
            cajas_grandes_recortes
        ) = detectar_sillas_en_recortes(
            imagen,
            contador_ids
        )

    candidatas_sillas = (
        sillas_completas + detecciones_recortes
    )

    sillas_seleccionadas, duplicados_sillas = (
        eliminar_duplicados(
            candidatas_sillas,
            umbral_contencion=0.82,
            umbral_iou=UMBRAL_IOU_SILLA
        )
    )

    # Numerar los objetos finales después de eliminar duplicados.
    personas_finales = [
        {
            "id": indice + 1,
            "id_modelo": d["id_modelo"],
            "confidence": d["confianza"],
            "box": d["box"],
            "origen": d["origen"]
        }
        for indice, d in enumerate(personas)
    ]

    sillas_finales = [
        {
            "id": indice + 1,
            "id_modelo": d["id_modelo"],
            "confidence": d["confianza"],
            "box": d["box"],
            "origen": d["origen"]
        }
        for indice, d in enumerate(sillas_seleccionadas)
    ]

    (
        ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_silla,
        sillas_sin_persona
    ) = asociar_personas_sillas(
        personas_finales,
        sillas_finales
    )

    total_personas = len(personas_finales)
    total_sillas = len(sillas_finales)

    libres = max(total_sillas - ocupadas, 0)

    estado = "vacio" if total_personas == 0 else "ocupado"

    ids_personas = {
        d["id_modelo"] for d in personas_finales
    }

    ids_sillas = {
        d["id_modelo"] for d in sillas_finales
    }

    diagnostico_completo = []

    for d in detecciones_completas:
        incluido = (
            d["id_modelo"] in ids_personas
            if d["clase"] == 0
            else d["id_modelo"] in ids_sillas
        )

        diagnostico_completo.append({
            "id_modelo": d["id_modelo"],
            "clase": "persona" if d["clase"] == 0 else "silla",
            "confianza": round(d["confianza"], 3),
            "incluida_en_conteo": incluido,
            "origen": d["origen"],
            "caja": [
                round(v, 1) for v in d["box"]
            ]
        })

    diagnostico_recortes = [
        {
            "id_modelo": d["id_modelo"],
            "confianza": round(d["confianza"], 3),
            "incluida_en_conteo": (
                d["id_modelo"] in ids_sillas
            ),
            "origen": d["origen"],
            "fraccion_area_imagen": d["fraccion_area_imagen"],
            "caja": [
                round(v, 1) for v in d["box"]
            ]
        }
        for d in detecciones_recortes
    ]

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": ocupadas,
        "libres": libres,
        "estado": estado,
        "diagnostico": {
            "recortes_activados": usar_recortes,
            "recortes": recortes_info,
            "detecciones_imagen_completa": diagnostico_completo,
            "detecciones_recortes": diagnostico_recortes,
            "personas_detectadas": [
                {
                    "id": d["id"],
                    "id_modelo": d["id_modelo"],
                    "confianza": round(d["confidence"], 3),
                    "caja": [
                        round(v, 1) for v in d["box"]
                    ]
                }
                for d in personas_finales
            ],
            "sillas_detectadas": [
                {
                    "id": d["id"],
                    "id_modelo": d["id_modelo"],
                    "confianza": round(d["confidence"], 3),
                    "origen": d["origen"],
                    "caja": [
                        round(v, 1) for v in d["box"]
                    ]
                }
                for d in sillas_finales
            ],
            "duplicados_descartados": (
                duplicados_personas + duplicados_sillas
            ),
            "cajas_grandes_descartadas": (
                cajas_grandes_completas
                + cajas_grandes_recortes
            ),
            "asignaciones": asignaciones,
            "coincidencias_evaluadas": coincidencias,
            "personas_sin_silla": personas_sin_silla,
            "sillas_sin_persona": sillas_sin_persona
        }
    }

    del detecciones_completas
    del detecciones_recortes
    del candidatas_sillas
    del personas
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