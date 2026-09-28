from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

app = FastAPI()

# Modelo pequeño para que consuma menos recursos
model = YOLO("yolo11n.pt")


@app.get("/")
def inicio():
    return {
        "mensaje": "Servidor de tutoria funcionando"
    }


@app.get("/health")
def health():
    return {
        "estado": "ok"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Leer la imagen recibida
    image_bytes = await file.read()

    # Guardarla temporalmente
    with open("temp.jpg", "wb") as f:
        f.write(image_bytes)

    # Ejecutar YOLO
    results = model("temp.jpg")

    people = 0
    chairs = 0

    # Contar personas y sillas
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            # COCO: 0 = person
            if class_id == 0:
                people += 1

            # COCO: 56 = chair
            elif class_id == 56:
                chairs += 1

    return {
        "personas": people,
        "sillas": chairs
    }
