import modal
import base64
import io
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# 1. ĐỊNH NGHĨA MÔI TRƯỜNG (IMAGE)
# Cài đặt đầy đủ các thư viện hệ thống (apt) và python (pip)
app_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libz-dev", 
        "libjpeg-dev", 
        "libpng-dev", 
        "libgl1",           # Sửa lỗi libGL.so.1 (OpenCV)
        "libglib2.0-0"      # Sửa lỗi glib (OpenCV)
    )
    .pip_install(
        "simple-lama-inpainting",
        "pillow",
        "fastapi",
        "pydantic",
        "accelerate",
        "torch"
    )
)

# 2. KHỞI TẠO APP VÀ WEB SERVER
app = modal.App("lama-cleaner-v2", image=app_image)
web_app = FastAPI()

# Mở cổng CORS để App StoryBrain có thể gọi vào mà không bị chặn
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. LỚP XỬ LÝ AI (MODEL CLASS)
@app.cls(
    gpu="T4",               # Dùng T4 cho rẻ, tốc độ LaMa trên T4 là cực nhanh
    timeout=300,
    scaledown_window=60     # Tự động tắt GPU sau 60s không dùng để tiết kiệm tiền
)
class LaMaModel:
    @modal.enter()
    def load_model(self):
        from simple_lama_inpainting import SimpleLama
        print("🚀 Đang khởi động mô hình LaMa...")
        self.lama = SimpleLama()

    @modal.method()
    def inpaint_process(self, image_bytes: bytes, mask_bytes: bytes):
        from PIL import Image
        
        # Đọc ảnh và mask từ bytes
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        # Chạy thuật toán LaMa
        result = self.lama(img, mask)
        
        # Chuyển kết quả về lại bytes
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()

# 4. ENDPOINT API CHO STORYBRAIN
@web_app.post("/img2img")
async def handle_request(request: Request):
    try:
        body = await request.json()
        
        # Lấy dữ liệu ảnh (hỗ trợ nhiều định dạng key mà app có thể gửi)
        img_data = body.get("image") or (body.get("init_images")[0] if body.get("init_images") else None)
        mask_data = body.get("mask") or body.get("mask_image")

        if not img_data or not mask_data:
            return {"error": "Thiếu dữ liệu ảnh hoặc mask"}

        # Hàm helper giải mã base64
        def b64_to_bytes(b64_str):
            if "," in b64_str: b64_str = b64_str.split(",")[-1]
            return base64.b64decode(b64_str)

        # Xử lý song song qua Modal GPU
        model = LaMaModel()
        res_bytes = await model.inpaint_process.remote.aio(
            b64_to_bytes(img_data), 
            b64_to_bytes(mask_data)
        )
        
        # Mã hóa kết quả trả về
        res_b64 = base64.b64encode(res_bytes).decode("utf-8")

        # Trả về format mà các app Stable Diffusion / StoryBrain mong đợi
        return {
            "images": [res_b64],
            "data": [{"b64_json": res_b64}]
        }

    except Exception as e:
        print(f"🚨 Lỗi hệ thống: {str(e)}")
        return {"error": str(e)}

# 5. KÍCH HOẠT WEB APP
@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
