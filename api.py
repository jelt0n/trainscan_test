from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import base64
import io
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Загружаем модель. 
# Лучше использовать .pt, если есть. Если только .onnx, укажи 'model/best.onnx'
# task='detect' нужен, если загружаешь onnx, для .pt он определяется сам
try:
    model = YOLO('model/best.pt') 
except:
    print("Файл .pt не найден, пробую загрузить .onnx")
    model = YOLO('model/best.onnx', task='detect')

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Читаем файл и конвертируем в PIL Image
        image_bytes = await file.read()
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
        except:
            return {"success": False, "error": "Некорректный файл изображения"}
        
        # 2. ПРЕДСКАЗАНИЕ (Ultralytics делает всё сама: ресайз, NMS, конфиденс)
        # conf=0.25 - порог уверенности
        # iou=0.45 - порог NMS
        results = model.predict(source=pil_image, conf=0.25, iou=0.45)
        
        result = results[0] # Берем первый (и единственный) результат
        
        # 3. Формируем JSON ответ (как в твоем старом коде)
        predictions = []
        
        # result.boxes содержит всё необходимое
        for box in result.boxes:
            # Координаты (xyxy)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] # Имена классов берутся прямо из модели!
            
            predictions.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'bbox_int': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': conf,
                'class': cls_id,
                'class_name': cls_name
            })

        result_img_bgr = result.plot() 
        
        _, buffer = cv2.imencode('.jpg', result_img_bgr)
        result_base64 = base64.b64encode(buffer).decode()
        
        orig_base64 = base64.b64encode(image_bytes).decode()
        
        return {
            "success": True,
            "predictions": predictions,
            "result_image": result_base64,
            "original_image": orig_base64,
            "count": len(predictions)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/test_model")
async def test_model():
    """Простой тест, что модель загружена"""
    try:
        # Создаем черный квадрат для теста
        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Прогоняем через модель
        results = model.predict(test_img, verbose=False)
        
        return {
            "model_loaded": True,
            "classes": model.names, # Список классов внутри модели
            "device": str(model.device)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Запуск API на http://localhost:8000")
    print(f"🎯 Классы модели: {model.names}")
    uvicorn.run(app, host="0.0.0.0", port=8000)