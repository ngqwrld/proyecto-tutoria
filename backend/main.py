from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO
import os
import tempfile

app = FastAPI()

# ============================================================
# MODELO
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# RUTAS
# ============================================================

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


# ============================================================
# COMPROBAR RELACIÓN PERSONA - SILLA
# ============================================================

def punto_en_silla(person_box, chair_box):

    px1, py1, px2, py2 = person_box
    cx1, cy1, cx2, cy2 = chair_box

    # Punto inferior central de la persona
    punto_x = (px1 + px2) / 2
    punto_y = py2

    # Dimensiones de la silla
    ancho = cx2 - cx1
    alto = cy2 - cy1

    # Margen horizontal y vertical
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
# PREDICT
# ============================================================

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

    temp_path = None

    try:

        # ----------------------------------------------------
        # GUARDAR IMAGEN TEMPORAL
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_file.write(image_bytes)
            temp_path = temp_file.name

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        results = model(
            temp_path,
            conf=0.30,
            verbose=False
        )

        people = []
        chairs = []

        # ----------------------------------------------------
        # PROCESAR DETECCIONES
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                coordinates = box.xyxy[0].tolist()

                # PERSONA
                if class_id == 0:

                    people.append({
                        "box": coordinates,
                        "confidence": confidence
                    })

                # SILLA
                elif class_id == 56:

                    chairs.append({
                        "box": coordinates,
                        "confidence": confidence
                    })

        # ----------------------------------------------------
        # DETERMINAR SILLAS OCUPADAS
        # ----------------------------------------------------

        occupied_chairs = 0

        for chair in chairs:

            chair_box = chair["box"]

            for person in people:

                person_box = person["box"]

                if punto_en_silla(
                    person_box,
                    chair_box
                ):

                    occupied_chairs += 1
                    break

        # ----------------------------------------------------
        # RESULTADOS
        # ----------------------------------------------------

        people_count = len(people)

        total_chairs = len(chairs)

        free_chairs = max(
            total_chairs - occupied_chairs,
            0
        )

        # ----------------------------------------------------
        # ESTADO DEL SALÓN
        # ----------------------------------------------------

        if people_count == 0:
            estado = "vacio"
        else:
            estado = "ocupado"

        # ----------------------------------------------------
        # RESPUESTA
        # ----------------------------------------------------

        return {
            "personas": people_count,
            "sillas": total_chairs,
            "ocupadas": occupied_chairs,
            "libres": free_chairs,
            "estado": estado
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)