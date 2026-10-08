from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import os
import tempfile

app = FastAPI()

# ============================================================
# MODELO YOLO
# ============================================================

# YOLO11s: modelo más preciso que YOLO11n
# Ultralytics lo descargará automáticamente si no existe.
model = YOLO("yolo11n.pt")


# ============================================================
# RUTAS BÁSICAS
# ============================================================

@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando",
        "modelo": "YOLO11s"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok",
        "modelo": "YOLO11s"
    }


# ============================================================
# COMPROBAR SI UNA PERSONA ESTÁ SOBRE UNA SILLA
# ============================================================

def punto_en_silla(person_box, chair_box):
    """
    Comprueba si la parte inferior de una persona
    está dentro de la zona de una silla.

    person_box = [x1, y1, x2, y2]
    chair_box = [x1, y1, x2, y2]
    """

    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = chair_box

    # Punto utilizado:
    # centro de la parte inferior de la persona
    punto_x = (px1 + px2) / 2
    punto_y = py2

    # Tamaño de la silla
    ancho = cx2 - cx1
    alto = cy2 - cy1

    # Márgenes para hacer la zona de la silla
    # ligeramente más tolerante
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


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Verificar que se haya enviado un archivo
    if not file:
        raise HTTPException(
            status_code=400,
            detail="No se recibió ninguna imagen"
        )

    # Leer imagen
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="La imagen está vacía"
        )

    # Archivo temporal
    temp_path = None

    try:

        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        # ====================================================
        # EJECUTAR YOLO11s
        # ====================================================

        results = model(
            temp_path,
            conf=0.25
        )

        people = []
        chairs = []

        # ====================================================
        # PROCESAR DETECCIONES
        # ====================================================

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                coordinates = box.xyxy[0].tolist()

                # --------------------------------------------
                # PERSONA
                # COCO class 0 = person
                # --------------------------------------------

                if class_id == 0:

                    people.append({
                        "box": coordinates,
                        "confidence": confidence
                    })

                # --------------------------------------------
                # SILLA
                # COCO class 56 = chair
                # --------------------------------------------

                elif class_id == 56:

                    chairs.append({
                        "box": coordinates,
                        "confidence": confidence
                    })

        # ====================================================
        # DETERMINAR SILLAS OCUPADAS
        # ====================================================

        occupied_chairs = 0

        for chair in chairs:

            chair_box = chair["box"]

            silla_ocupada = False

            for person in people:

                person_box = person["box"]

                if punto_en_silla(
                    person_box,
                    chair_box
                ):
                    silla_ocupada = True
                    break

            if silla_ocupada:
                occupied_chairs += 1

        # ====================================================
        # RESULTADOS
        # ====================================================

        total_chairs = len(chairs)

        free_chairs = total_chairs - occupied_chairs

        # Para nuestro proyecto:
        # contamos como personas a quienes realmente
        # están ocupando una silla.
        people_count = occupied_chairs

        # Evitar valores negativos por seguridad
        if free_chairs < 0:
            free_chairs = 0

        # ====================================================
        # RESPUESTA
        # ====================================================

        return {
            "personas": people_count,
            "sillas": total_chairs,
            "ocupadas": occupied_chairs,
            "libres": free_chairs
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )

    finally:

        # Eliminar archivo temporal
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)