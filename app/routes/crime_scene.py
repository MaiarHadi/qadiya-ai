
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile
)

from app.services import image_processing as ip


# إنشاء Router خاص بمعالجة صور مسرح الجريمة
router = APIRouter(
    prefix="/api/crime-scene",
    tags=["Crime Scene"]
)

# الحد الأقصى لحجم الصورة هو 10 ميجابايت
MAX_FILE_SIZE = 10 * 1024 * 1024


@router.post("/crop")
async def crop_region(
    image: UploadFile = File(...),
    x: float = Form(..., ge=0, le=1),
    y: float = Form(..., ge=0, le=1),
    w: float = Form(..., gt=0, le=1),
    h: float = Form(..., gt=0, le=1),
):

    # قراءة بيانات الصورة القادمة من Node.js
    # مع تحديد أقصى كمية يمكن قراءتها
    data = await image.read(MAX_FILE_SIZE + 1)

    # التأكد من أن حجم الصورة لا يتجاوز الحد المسموح
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image is larger than 10 MB"
        )

    # التأكد من أن الصورة تحتوي على بيانات
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Image file is empty"
        )

    try:
        # تحويل بيانات الصورة إلى مصفوفة OpenCV
        img = ip.decode_image(data)

        # قص المنطقة التي حددها المستخدم
        cropped = ip.crop_image(
            img, x, y, w, h
        )

        # تحويل الصورة المقصوصة إلى PNG
        png_bytes = ip.encode_png(cropped)

    # التعامل مع أخطاء معالجة الصور والإحداثيات
    except ip.ImageProcessingError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        ) from e

    # استخراج أبعاد الصورة المقصوصة
    crop_h, crop_w = cropped.shape[:2]

    # إرجاع الصورة المقصوصة إلى Node.js
    # مع عرضها وارتفاعها في Headers
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Crop-Width": str(crop_w),
            "X-Crop-Height": str(crop_h)
        }
    )
