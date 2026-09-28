from fastapi import FastAPI

app = FastAPI()


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
