
from fastapi import FastAPI


from app.routes import crime_scene, image, ocr as ocr_routes
from app.services.ocr import start_preload

app = FastAPI(
    title="Qadiya AI Service",
    description="AI and Computer Vision services for Qadiya",
    version="1.0.0"
)

from fastapi import FastAPI

from app.routes import crime_scene, image

app = FastAPI(title="QADIYA AI Service")

app.include_router(crime_scene.router)

app.include_router(image.router)


app.include_router(ocr_routes.router)
start_preload()

@app.get("/health")
def health():
    return {"status": "ok"}
