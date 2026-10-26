
from pathlib import Path
import sys
 
import cv2
import numpy as np
import requests
 
API_URL = "http://127.0.0.1:8000/api/crime-scene/crop"
IMAGE_PATH = Path(__file__).with_name("crime_scene_sample.png")
OUTPUT_PATH = Path(__file__).with_name("cropped_result.png")
WINDOW_NAME = "Qadiya - Crime Scene Selection"
RESULT_WINDOW = "Qadiya - API Crop Result"
VIEW_W, VIEW_H = 920, 660
PANEL_W = 330
MIN_SIZE = 2  # الحد الأدنى لحجم التحديد على الصورة الأصلية
HANDLE_RADIUS = 12
 
 
class SelectionTool:
    def __init__(self, image):
        self.image = image
        self.img_h, self.img_w = image.shape[:2]
        self.base_scale = min(VIEW_W / self.img_w, VIEW_H / self.img_h)
        self.scale = self.base_scale
        self.offset_x = (VIEW_W - self.img_w * self.scale) / 2
        self.offset_y = (VIEW_H - self.img_h * self.scale) / 2
        self.box = None  # [left, top, right, bottom] بالبكسل الأصلي
        self.mode = None
        self.corner = None
        self.start = (0.0, 0.0)
        self.initial_box = None
        self.pan_origin = (0, 0)
        self.pan_offset = (0.0, 0.0)
 
    def clamp_point(self, ix, iy):
        return max(0.0, min(ix, float(self.img_w))), max(0.0, min(iy, float(self.img_h)))
 
    def screen_to_image(self, sx, sy):
        return self.clamp_point((sx - self.offset_x) / self.scale,
                                (sy - self.offset_y) / self.scale)
 
    def image_to_screen(self, ix, iy):
        return int(round(ix * self.scale + self.offset_x)), int(round(iy * self.scale + self.offset_y))
 
    def corners(self):
        if self.box is None:
            return {}
        l, t, r, b = self.box
        return {"tl": (l, t), "tr": (r, t), "bl": (l, b), "br": (r, b)}
 
    def normalized(self):
        if self.box is None:
            return None
        l, t, r, b = self.box
        if r - l < MIN_SIZE or b - t < MIN_SIZE:
            return None
        return {"x": l / self.img_w, "y": t / self.img_h,
                "w": (r - l) / self.img_w, "h": (b - t) / self.img_h}
 
    def live_crop(self):
        if self.normalized() is None:
            return None
        l, t, r, b = self.box
        x1 = max(0, min(round(l), self.img_w))
        y1 = max(0, min(round(t), self.img_h))
        x2 = max(0, min(round(r), self.img_w))
        y2 = max(0, min(round(b), self.img_h))
        if x2 <= x1 or y2 <= y1:
            return None
        return self.image[y1:y2, x1:x2]
 
    def on_mouse(self, event, sx, sy, flags, userdata):
        inside = 0 <= sx < VIEW_W and 0 <= sy < VIEW_H
 
        if event == cv2.EVENT_MOUSEWHEEL and inside:
            # اتجاه العجلة مخزّن بأعلى 16 bit من flags كرقم بإشارة
            wheel_delta = (flags >> 16) & 0xFFFF
            if wheel_delta >= 0x8000:
                wheel_delta -= 0x10000
            if wheel_delta == 0:
                return
 
            multiplier = 1.2 if wheel_delta > 0 else 1 / 1.2
            new_scale = max(self.base_scale, min(self.base_scale * 8, self.scale * multiplier))
 
            # إبقاء النقطة الموجودة تحت مؤشر الماوس في نفس مكانها أثناء التكبير
            image_x = (sx - self.offset_x) / self.scale
            image_y = (sy - self.offset_y) / self.scale
            self.offset_x = sx - image_x * new_scale
            self.offset_y = sy - image_y * new_scale
            self.scale = new_scale
            return
 
        if event == cv2.EVENT_RBUTTONDOWN and inside:
            self.mode = "pan"
            self.pan_origin = (sx, sy)
            self.pan_offset = (self.offset_x, self.offset_y)
            return
        if event == cv2.EVENT_RBUTTONUP and self.mode == "pan":
            self.mode = None
            return
        if event == cv2.EVENT_MOUSEMOVE and self.mode == "pan":
            self.offset_x = self.pan_offset[0] + sx - self.pan_origin[0]
            self.offset_y = self.pan_offset[1] + sy - self.pan_origin[1]
            return
 
        if event == cv2.EVENT_LBUTTONDOWN and inside:
            # ما بنبدأ تحديدًا جديدًا خارج مساحة الصورة الفعلية
            raw_x = (sx - self.offset_x) / self.scale
            raw_y = (sy - self.offset_y) / self.scale
            ix, iy = self.clamp_point(raw_x, raw_y)
            self.start = (ix, iy)
            self.initial_box = self.box.copy() if self.box else None
            if self.box:
                for name, (cx, cy) in self.corners().items():
                    px, py = self.image_to_screen(cx, cy)
                    if abs(sx - px) <= HANDLE_RADIUS and abs(sy - py) <= HANDLE_RADIUS:
                        self.mode, self.corner = "resize", name
                        return
                l, t, r, b = self.box
                if l <= ix <= r and t <= iy <= b:
                    self.mode = "move"
                    return
            if not (0 <= raw_x <= self.img_w and 0 <= raw_y <= self.img_h):
                return
            self.box = [ix, iy, ix, iy]
            self.mode = "draw"
            return
 
        if event == cv2.EVENT_MOUSEMOVE and self.mode in ("draw", "move", "resize"):
            ix, iy = self.screen_to_image(sx, sy)
            start_x, start_y = self.start
            if self.mode == "draw":
                self.box = [min(start_x, ix), min(start_y, iy),
                            max(start_x, ix), max(start_y, iy)]
            elif self.mode == "move":
                l, t, r, b = self.initial_box
                dx = max(-l, min(ix - start_x, self.img_w - r))
                dy = max(-t, min(iy - start_y, self.img_h - b))
                self.box = [l + dx, t + dy, r + dx, b + dy]
            elif self.mode == "resize":
                l, t, r, b = self.initial_box
                if self.corner.endswith("l"):
                    l = min(ix, r - MIN_SIZE)
                else:
                    r = max(ix, l + MIN_SIZE)
                if self.corner.startswith("t"):
                    t = min(iy, b - MIN_SIZE)
                else:
                    b = max(iy, t + MIN_SIZE)
                self.box = [max(0, l), max(0, t), min(self.img_w, r), min(self.img_h, b)]
            return
 
        if event == cv2.EVENT_LBUTTONUP and self.mode in ("draw", "move", "resize"):
            if self.normalized() is None:
                self.box = None
            self.mode = None
            self.corner = None
 
    @staticmethod
    def label(canvas, text, x, y, size=0.54, color=(235, 235, 235)):
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    size, color, 1, cv2.LINE_AA)
 
    def draw(self):
        canvas = np.full((VIEW_H, VIEW_W + PANEL_W, 3), (25, 27, 33), dtype=np.uint8)
        # صورة العرض تتغير مع Zoom/Pan؛ إحداثيات المستطيل تظل أصلية دومًا
        matrix = np.array([[self.scale, 0, self.offset_x],
                           [0, self.scale, self.offset_y]], dtype=np.float32)
        view = cv2.warpAffine(self.image, matrix, (VIEW_W, VIEW_H),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(30, 32, 37))
        canvas[:, :VIEW_W] = view
 
        if self.box:
            l, t, r, b = self.box
            p1 = self.image_to_screen(l, t)
            p2 = self.image_to_screen(r, b)
            cv2.rectangle(canvas, p1, p2, (0, 220, 255), 2)
            for cx, cy in self.corners().values():
                px, py = self.image_to_screen(cx, cy)
                cv2.rectangle(canvas, (px - 5, py - 5), (px + 5, py + 5),
                              (0, 220, 255), -1)
 
        p = VIEW_W
        cv2.rectangle(canvas, (p, 0), (p + PANEL_W, VIEW_H), (39, 42, 50), -1)
        self.label(canvas, "QADIYA - LIVE PREVIEW", p + 17, 30, 0.63, (90, 230, 255))
        self.label(canvas, "Selected region (local preview)", p + 17, 58, 0.45)
        preview = self.live_crop()
        if preview is not None:
            ph, pw = preview.shape[:2]
            s = min((PANEL_W - 28) / pw, 280 / ph)
            fitted = cv2.resize(preview, (max(1, round(pw * s)), max(1, round(ph * s))))
            fh, fw = fitted.shape[:2]
            xx, yy = p + (PANEL_W - fw) // 2, 85 + (280 - fh) // 2
            canvas[yy:yy + fh, xx:xx + fw] = fitted
            self.label(canvas, f"Preview: {pw} x {ph} px", p + 17, 392, 0.5)
            coords = self.normalized()
            for row, (key, val) in enumerate(coords.items()):
                self.label(canvas, f"{key} = {val:.5f}", p + 17, 420 + 26 * row, 0.54)
        else:
            self.label(canvas, "Draw a rectangle on the image", p + 18, 210, 0.49)
 
        cv2.line(canvas, (p + 15, 535), (p + PANEL_W - 15, 535), (90, 90, 90), 1)
        for row, caption in enumerate([
            "Wheel: Zoom in/out",
            "Right drag: Pan image",
            "Left drag: New selection",
            "Inside box: Move selection",
            "Yellow corners: Resize",
            "C: Crop via API    R: Reset",
            "0: Reset zoom      Q: Quit",
        ]):
            self.label(canvas, caption, p + 17, 552 + 16 * row, 0.43)
        return canvas
 
 
def main():
    image_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else IMAGE_PATH
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Image not found or invalid: {image_path}")
        return
 
    tool = SelectionTool(image)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, tool.on_mouse)
    print("Mouse wheel = zoom | Right drag = pan | Left drag = select/move/resize")
    print("C = crop via FastAPI | R = clear selection | 0 = reset zoom | Q = quit")
 
    while True:
        cv2.imshow(WINDOW_NAME, tool.draw())
        key = cv2.waitKey(25) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            tool.box = None
        if key == ord("0"):
            tool.scale = tool.base_scale
            tool.offset_x = (VIEW_W - tool.img_w * tool.scale) / 2
            tool.offset_y = (VIEW_H - tool.img_h * tool.scale) / 2
        if key == ord("c"):
            coords = tool.normalized()
            if coords is None:
                print("Select an area first.")
                continue
            print("Normalized coordinates:", coords)
            response = None
            try:
                with image_path.open("rb") as handle:
                    response = requests.post(
                        API_URL,
                        files={"image": (image_path.name, handle, "application/octet-stream")},
                        data={key: f"{val:.10f}" for key, val in coords.items()},
                        timeout=30,
                    )
                response.raise_for_status()
            except requests.RequestException as error:
                print("Crop API request failed:", error)
                if response is not None:
                    print("Server response:", response.text[:700])
                continue
 
            OUTPUT_PATH.write_bytes(response.content)
            arr = np.frombuffer(response.content, dtype=np.uint8)
            result = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if result is None:
                print("API did not return a valid PNG image.")
                continue
            rh, rw = result.shape[:2]
            fitted_scale = min(1.0, 800 / rw, 650 / rh)
            display = cv2.resize(result, (max(1, round(rw * fitted_scale)),
                                          max(1, round(rh * fitted_scale))))
            cv2.imshow(RESULT_WINDOW, display)
            print(f"Saved API result: {OUTPUT_PATH}")
            print("Crop size:", response.headers.get("X-Crop-Width"), "x",
                  response.headers.get("X-Crop-Height"))
 
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()