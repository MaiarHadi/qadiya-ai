from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile

from app.services import image_processing as ip
from app.services.enhance import enhance_image

router = APIRouter(prefix="/api/image", tags=["Image Analysis"])

MAX_FILE_SIZE = 10 * 1024 * 1024  


@router.post("/enhance")
async def enhance(
    image: UploadFile = File(...),

    brightness: int = Form(0, ge=-100, le=100),
    contrast: float = Form(1.0, ge=0.5, le=3.0),
    grayscale: bool = Form(False),
     
    sharpen: float = Form(0.0, ge=0.0, le=3.0),
    denoise: int = Form(0, ge=0, le=30),
    clahe: bool = Form(False),
    upscale: int = Form(1, ge=1, le=4),
):
    data = await image.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB")

    try:
        img = ip.decode_image(data)
        result = enhance_image(
            img,
            brightness=brightness,
            contrast=contrast,
            grayscale=grayscale,
            sharpen_amount=sharpen,
            denoise_strength=denoise,
            use_clahe=clahe,
            upscale_factor=upscale,
        )
        png = ip.encode_png(result)
    except ip.ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    h, w = result.shape[:2]
    return Response(
        content=png,
        media_type="image/png",
        headers={"X-Image-Width": str(w), "X-Image-Height": str(h)},
    )