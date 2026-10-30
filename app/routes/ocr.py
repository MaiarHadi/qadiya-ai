from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services import image_processing as ip
from app.services import ocr

router = APIRouter(prefix="/api/image", tags=["OCR"])

MAX_FILE_SIZE = 10 * 1024 * 1024  
@router.post("/ocr")
def read_image_text(
    image: UploadFile = File(...),
 
    lang: Literal["auto", "en", "ar"] = Form("auto"),
  
    x: Optional[float] = Form(None, ge=0, le=1),
    y: Optional[float] = Form(None, ge=0, le=1),
    w: Optional[float] = Form(None, gt=0, le=1),
    h: Optional[float] = Form(None, gt=0, le=1),
):
    data = image.file.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB")

    box = (x, y, w, h)
    if any(v is not None for v in box) and any(v is None for v in box):
        raise HTTPException(status_code=422, detail="Send all of x, y, w, h or none of them")

    try:
        img = ip.decode_image(data)
        if x is not None:
            img = ocr.crop_normalized(img, x, y, w, h)
        result = ocr.read_text(img, lang)
    except ip.ImageProcessingError as e:
        raise HTTPException(status_code=400, detail=str(e))

    height, width = img.shape[:2]
    return {**result, "width": width, "height": height}