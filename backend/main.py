
import os

# Limitar hilos para reducir consumo de CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import asyncio
import gc
import json
import logging
import time
import uuid
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import onnxruntime as ort

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

MAX_IMAGE_BYTES = 8 * 1024 * 1024
INPUT_SIZE = 320
CONF_PERSONA = 0.30
NMS_THRESHOLD = 0.45

VERDE_MIN = np.array([25, 40, 35], dtype=np.uint8)
VERDE_MAX = np.array([95, 255, 255], dtype=np.uint8)

prediction_lock = asyncio.Lock()

MODEL_PATH = Path(__file__).resolve().parent / "yolo11n.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"No se encontró el modelo ONNX: {MODEL_PATH}"
    )


def leer_memoria_proceso_mb():
    """
    Lee VmRSS (RAM residente actual) y VmHWM
    (máximo residente desde el inicio del proceso).

    Funciona en Linux, como el entorno de Render.
    Si /proc no está disponible, devuelve None.
    """

    datos = {
        "rss_mb": None,
        "hwm_mb": None
    }

    try:
        with open("/proc/self/status", "r", encoding="utf-8") as archivo:
            for linea in archivo:
                if linea.startswith("VmRSS:"):
                    kb = int(linea.split()[1])
                    datos["rss_mb"] = round(kb / 1024, 2)

                elif linea.startswith("VmHWM:"):
                    kb = int(linea.split()[1])
                    datos["hwm_mb"] = round(kb / 1024, 2)

                if (
                    datos["rss_mb"] is not None
                    and datos["hwm_mb"] is not None
                ):
                    break

    except (OSError, ValueError, IndexError):
        pass

    return datos


async def muestrear_memoria(stop_event, tracker):
    """
    Muestrea la RAM del proceso cada 100 ms durante
    una solicitud de análisis.

    El pico es una estimación muestreada: un pico muy
    breve podría ocurrir entre dos mediciones.
    """

    while not stop_event.is_set():
        memoria = leer_memoria_proceso_mb()
        rss = memoria["rss_mb"]

        if rss is not None:
            actual = tracker.get("pico_rss_mb")

            if actual is None or rss > actual:
                tracker["pico_rss_mb"] = rss

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=0.1
            )
        except asyncio.TimeoutError:
            pass


# Medir la carga del modelo al arrancar
inicio_carga = time.perf_counter()

options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_names = [item.name for item in session.get_outputs()]

CARGA_MODELO_MS = round(
    (time.perf_counter() - inicio_carga) * 1000, 2
)

MEMORIA_INICIO = leer_memoria_proceso_mb()

logger.info(json.dumps({
    "evento": "modelo_onnx_cargado",
    "modelo": MODEL_PATH.name,
    "carga_modelo_ms": CARGA_MODELO_MS,
    "memoria_inicio_mb": MEMORIA_INICIO,
    "input_shape": session.get_inputs()[0].shape,
    "output_shapes": [
        item.shape for item in session.get_outputs()
    ]
}))


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando",
        "modelo": "YOLO11n ONNX + OpenCV",
        "carga_modelo_ms": CARGA_MODELO_MS
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": "YOLO11n ONNX + OpenCV"
    }


def preparar_entrada(imagen_rgb):
    """Redimensiona y rellena la imagen hasta 320x320."""

    inicio = time.perf_counter()

    alto, ancho = imagen_rgb.shape[:2]
    escala = min(INPUT_SIZE / ancho, INPUT_SIZE / alto)

    nuevo_ancho = max(1, int(round(ancho * escala)))
    nuevo_alto = max(1, int(round(alto * escala)))

    redimensionada = cv2.resize(
        imagen_rgb,
        (nuevo_ancho, nuevo_alto),
        interpolation=cv2.INTER_LINEAR
    )

    izquierda = (INPUT_SIZE - nuevo_ancho) // 2
    arriba = (INPUT_SIZE - nuevo_alto) // 2
    derecha = INPUT_SIZE - nuevo_ancho - izquierda
    abajo = INPUT_SIZE - nuevo_alto - arriba

    entrada = cv2.copyMakeBorder(
        redimensionada,
        arriba,
        abajo,
        izquierda,
        derecha,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    entrada = entrada.astype(np.float32) / 255.0
    entrada = np.transpose(entrada, (2, 0, 1))
    entrada = np.expand_dims(entrada, axis=0)
    entrada = np.ascontiguousarray(entrada)

    escala_x = nuevo_ancho / ancho
    escala_y = nuevo_alto / alto

    tiempo_ms = round(
        (time.perf_counter() - inicio) * 1000, 2
    )

    return (
        entrada,
        escala_x,
        escala_y,
        izquierda,
        arriba,
        tiempo_ms
    )


def detectar_personas(imagen):
    """Ejecuta ONNX Runtime y extrae las personas detectadas."""

    with imagen.convert("RGB") as rgb:
        imagen_rgb = np.array(rgb)

    alto_original, ancho_original = imagen_rgb.shape[:2]

    (
        entrada,
        escala_x,
        escala_y,
        izquierda,
        arriba,
        preparacion_ms
    ) = preparar_entrada(imagen_rgb)

    inicio_inferencia = time.perf_counter()

    outputs = session.run(
        output_names,
        {input_name: entrada}
    )

    inferencia_ms = round(
        (time.perf_counter() - inicio_inferencia) * 1000, 2
    )

    inicio_postproceso = time.perf_counter()

    predicciones = np.asarray(outputs[0])

    if predicciones.ndim == 3:
        predicciones = predicciones[0]

    # YOLO11n COCO: 4 coordenadas y 80 clases.
    if predicciones.shape[0] < predicciones.shape[1]:
        predicciones = predicciones.T

    if predicciones.shape[1] != 84:
        raise RuntimeError(
            f"Formato de salida ONNX inesperado: {predicciones.shape}"
        )

    # Clase 0 = persona.
    puntuaciones = predicciones[:, 4]
    indices = np.flatnonzero(
        puntuaciones >= CONF_PERSONA
    )

    cajas_nms = []
    confianzas = []
    cajas_originales = []

    for indice in indices:
        centro_x, centro_y, ancho, alto = (
            predicciones[indice, :4].astype(float)
        )

        x1_modelo = centro_x - ancho / 2
        y1_modelo = centro_y - alto / 2
        x2_modelo = centro_x + ancho / 2
        y2_modelo = centro_y + alto / 2

        cajas_nms.append([
            int(round(x1_modelo)),
            int(round(y1_modelo)),
            max(1, int(round(ancho))),
            max(1, int(round(alto)))
        ])

        confianzas.append(float(puntuaciones[indice]))

        x1 = (x1_modelo - izquierda) / escala_x
        y1 = (y1_modelo - arriba) / escala_y
        x2 = (x2_modelo - izquierda) / escala_x
        y2 = (y2_modelo - arriba) / escala_y

        cajas_originales.append([
            float(np.clip(x1, 0, ancho_original)),
            float(np.clip(y1, 0, alto_original)),
            float(np.clip(x2, 0, ancho_original)),
            float(np.clip(y2, 0, alto_original))
        ])

    personas = []

    if cajas_nms:
        indices_nms = cv2.dnn.NMSBoxes(
            cajas_nms,
            confianzas,
            CONF_PERSONA,
            NMS_THRESHOLD
        )

        if indices_nms is not None and len(indices_nms) > 0:
            for indice in np.asarray(indices_nms).reshape(-1):
                x1, y1, x2, y2 = cajas_originales[int(indice)]

                if x2 <= x1 or y2 <= y1:
                    continue

                personas.append({
                    "id": len(personas) + 1,
                    "confianza": round(
                        confianzas[int(indice)], 4
                    ),
                    "box": [x1, y1, x2, y2]
                })

    postproceso_ms = round(
        (time.perf_counter() - inicio_postproceso) * 1000, 2
    )

    tiempos = {
        "preparacion_entrada_ms": preparacion_ms,
        "yolo_inferencia_ms": inferencia_ms,
        "yolo_postproceso_ms": postproceso_ms,
        "yolo_total_ms": round(
            preparacion_ms + inferencia_ms + postproceso_ms, 2
        )
    }

    del outputs, predicciones, entrada, imagen_rgb
    gc.collect()

    return personas, tiempos


def detectar_sillas_verdes(imagen):
    """Detecta componentes de color verde usando OpenCV."""

    inicio = time.perf_counter()

    with imagen.convert("RGB") as rgb:
        imagen_rgb = np.array(rgb)

    alto, ancho = imagen_rgb.shape[:2]
    area_imagen = max(ancho * alto, 1)

    imagen_hsv = cv2.cvtColor(
        imagen_rgb,
        cv2.COLOR_RGB2HSV
    )

    mascara = cv2.inRange(
        imagen_hsv,
        VERDE_MIN,
        VERDE_MAX
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8)
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8)
    )

    cantidad, etiquetas, estadisticas, centroides = (
        cv2.connectedComponentsWithStats(
            mascara,
            connectivity=8
        )
    )

    area_minima = max(
        120,
        int(area_imagen * 0.0008)
    )

    sillas = []
    descartados = []

    for indice in range(1, cantidad):
        x = int(estadisticas[indice, cv2.CC_STAT_LEFT])
        y = int(estadisticas[indice, cv2.CC_STAT_TOP])
        w = int(estadisticas[indice, cv2.CC_STAT_WIDTH])
        h = int(estadisticas[indice, cv2.CC_STAT_HEIGHT])
        area = int(estadisticas[indice, cv2.CC_STAT_AREA])

        cx, cy = centroides[indice]

        if area < area_minima or w < 8 or h < 8:
            descartados.append({
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
                round(float(cx), 1),
                round(float(cy), 1)
            ]
        })

    diagnostico = {
        "componentes_verdes_totales": cantidad - 1,
        "area_minima_pixeles": area_minima,
        "componentes_descartados": descartados
    }

    tiempo_ms = round(
        (time.perf_counter() - inicio) * 1000, 2
    )

    del imagen_rgb, imagen_hsv, mascara, etiquetas

    return sillas, diagnostico, tiempo_ms


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

    zona_inferior = [
        px1,
        py1 + alto_persona * 0.35,
        px2,
        py2
    ]

    inter_total = interseccion_cajas(
        persona_box,
        silla_expandida
    )

    inter_inferior = interseccion_cajas(
        zona_inferior,
        silla_expandida
    )

    area_persona = max(area_caja(persona_box), 1)
    area_silla = max(area_caja(silla_expandida), 1)

    return max(
        inter_total / min(area_persona, area_silla),
        inter_inferior / area_silla
    )


def asociar_personas_sillas(personas, sillas):
    candidatos = []

    for persona in personas:
        for silla in sillas:
            puntuacion = relacion_persona_silla(
                persona["box"],
                silla["box"]
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

    return (
        len(sillas_asignadas),
        asignaciones,
        candidatos,
        [
            p["id"] for p in personas
            if p["id"] not in personas_asignadas
        ],
        [
            s["id"] for s in sillas
            if s["id"] not in sillas_asignadas
        ]
    )


def analizar_imagen(imagen):
    inicio_analisis = time.perf_counter()

    memoria_inicio = leer_memoria_proceso_mb()

    personas, tiempos_yolo = detectar_personas(imagen)

    memoria_despues_yolo = leer_memoria_proceso_mb()

    sillas, diagnostico_verde, opencv_ms = (
        detectar_sillas_verdes(imagen)
    )

    memoria_despues_opencv = leer_memoria_proceso_mb()

    inicio_asociacion = time.perf_counter()

    (
        ocupadas,
        asignaciones,
        coincidencias,
        personas_sin_silla,
        sillas_sin_persona
    ) = asociar_personas_sillas(personas, sillas)

    asociacion_ms = round(
        (time.perf_counter() - inicio_asociacion) * 1000,
        2
    )

    memoria_despues_asociacion = leer_memoria_proceso_mb()

    total_personas = len(personas)
    total_sillas = len(sillas)
    libres = max(total_sillas - ocupadas, 0)

    tiempos = dict(tiempos_yolo)
    tiempos["opencv_sillas_ms"] = opencv_ms
    tiempos["asociacion_ms"] = asociacion_ms
    tiempos["analisis_total_ms"] = round(
        (time.perf_counter() - inicio_analisis) * 1000, 2
    )

    respuesta = {
        "personas": total_personas,
        "sillas": total_sillas,
        "ocupadas": ocupadas,
        "libres": libres,
        "estado": "vacio" if total_personas == 0 else "ocupado",
        "diagnostico": {
            "metodo_personas": "YOLO11n ONNX Runtime",
            "metodo_sillas": "OpenCV HSV verde",
            "inferencias_yolo": 1,
            "personas_detectadas": [
                {
                    "id": p["id"],
                    "confianza": p["confianza"],
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
        "tiempos_ms": tiempos,
        "memoria_mb": {
            "antes_yolo_rss": memoria_inicio["rss_mb"],
            "despues_yolo_rss": memoria_despues_yolo["rss_mb"],
            "despues_opencv_rss": memoria_despues_opencv["rss_mb"],
            "despues_asociacion_rss": memoria_despues_asociacion["rss_mb"]
        }
    }

    del personas, sillas
    gc.collect()

    memoria_final = leer_memoria_proceso_mb()
    respuesta["memoria_mb"]["despues_gc_rss"] = memoria_final["rss_mb"]

    return respuesta


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    inicio_api = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]

    imagen = None
    image_bytes = b""
    stop_event = None
    sampler_task = None
    tracker = {"pico_rss_mb": None}

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
            (time.perf_counter() - inicio_decodificacion) * 1000,
            2
        )

        memoria_antes = leer_memoria_proceso_mb()
        tracker["pico_rss_mb"] = memoria_antes["rss_mb"]

        inicio_espera = time.perf_counter()

        async with prediction_lock:
            espera_lock_ms = round(
                (time.perf_counter() - inicio_espera) * 1000, 2
            )

            # Muestrear RAM solo durante el análisis de esta petición.
            stop_event = asyncio.Event()
            sampler_task = asyncio.create_task(
                muestrear_memoria(stop_event, tracker)
            )

            try:
                respuesta = await asyncio.to_thread(
                    analizar_imagen,
                    imagen
                )
            finally:
                stop_event.set()
                await sampler_task
                sampler_task = None

        memoria_despues = leer_memoria_proceso_mb()

        tiempos = respuesta["tiempos_ms"]
        tiempos["lectura_bytes_ms"] = lectura_ms
        tiempos["decodificacion_imagen_ms"] = decodificacion_ms
        tiempos["espera_turno_ms"] = espera_lock_ms
        tiempos["api_total_ms"] = round(
            (time.perf_counter() - inicio_api) * 1000, 2
        )

        respuesta["memoria_mb"].update({
            "antes_analisis_rss": memoria_antes["rss_mb"],
            "despues_analisis_rss": memoria_despues["rss_mb"],
            "pico_muestreado_solicitud_rss": tracker["pico_rss_mb"],
            "pico_proceso_desde_inicio_hwm": memoria_despues["hwm_mb"]
        })

        logger.info(json.dumps({
            "evento": "predict_completado",
            "request_id": request_id,
            "tiempos_ms": tiempos,
            "memoria_mb": respuesta["memoria_mb"],
            "personas": respuesta["personas"],
            "sillas": respuesta["sillas"]
        }))

        return respuesta

    except HTTPException as error:
        logger.warning(json.dumps({
            "evento": "predict_http_error",
            "request_id": request_id,
            "status_code": error.status_code
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
            detail="Error procesando la imagen. Revisa los logs."
        )

    finally:
        if stop_event is not None:
            stop_event.set()

        if sampler_task is not None:
            try:
                await sampler_task
            except Exception:
                pass

        if imagen is not None:
            imagen.close()

        image_bytes = b""
        await file.close()
        gc.collect()
