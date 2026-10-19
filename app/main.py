
from fastapi import FastAPI

app = FastAPI(
    title="Qadiya AI Service",
    description="AI and Computer Vision services for Qadiya",
    version="1.0.0"
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Qadiya AI"
    }
