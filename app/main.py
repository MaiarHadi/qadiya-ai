
from fastapi import FastAPI

app = FastAPI(
    title="Qadiya AI Service",
    description="AI and Computer Vision services for Qadiya",
    version="1.0.0"
)

from fastapi import FastAPI

from app.routes import crime_scene

app = FastAPI(title="QADIYA AI Service")

app.include_router(crime_scene.router)


@app.get("/health")
def health():
    return {"status": "ok"}