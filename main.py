from ultralytics import YOLO
import os
import torch
import time
import traceback

class YOLOTrainer:
    def __init__(self):
        self.project_dir = "C:/Users/aleks/Team9/Team9/Team9/trainScan/TrainScan"
        self.model = None
        self.device = None
        
    def setup(self):
        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = "cpu"

        try:
            self.model = YOLO("yolo11n.pt")
                
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
        
        return True
    
    def train(self):   
        start_time = time.time()
        try:
            train_args = {
                'data': os.path.join(self.project_dir, "conf.yaml"),
                'epochs': 10,
                'imgsz': 640,
                'batch': 16 if self.device != "cpu" else 8,
                'device': self.device,
                'workers': 4 if self.device != "cpu" else 0,
                'patience': 30,
                'save': True,
                'save_period': 1,
                'project': os.path.join(self.project_dir, "runs/final_train"),
                'name': 'exp_final',
                'exist_ok': True,
                'verbose': True,
                'seed': 42,
                'optimizer': 'AdamW',
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'mosaic': 1.0,
                'mixup': 0.2,
                'copy_paste': 0.1,
                'fliplr': 0.5,
                'flipud': 0.0,
                'degrees': 10.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,

                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'label_smoothing': 0.0,
                'dropout': 0.0,

                'close_mosaic': 10,
                'amp': True,
                'overlap_mask': True,
                'mask_ratio': 4,
                'nbs': 64,
                'single_cls': False,
                'plots': True,
                'rect': False,
                'cos_lr': True,
                'cache': True,
                'resume': False,
            }

            results = self.model.train(**train_args)
            
            training_time = time.time() - start_time
            print(f"\n{training_time/60:.1f} минут")
            
            return results
            
        except Exception as e:
            print(f"Ошибка при обучении: {e}")
            traceback.print_exc()
            return None
    
    def validate(self):
        try:
            best_model_path = os.path.join(
                self.project_dir, 
                "runs/final_train/exp_final/weights/best.pt"
            )
            
            if os.path.exists(best_model_path):
                best_model = YOLO(best_model_path)

                metrics = best_model.val(
                    data=os.path.join(self.project_dir, "conf.yaml"),
                    device=self.device,
                    conf=0.25,
                    iou=0.45,
                    plots=True,
                    save_json=True,
                    save_hybrid=True,
                )         
                return metrics
            
            else:
                print("⚠️ Лучшая модель не найдена")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка при валидации: {e}")
            return None
    
    def export_model(self):
        try:
            export_path = os.path.join(
                self.project_dir,
                "runs/final_train/exp_final/weights/best"
            )

            self.model.export(
                format="onnx",
                imgsz=640,
                simplify=True,
                opset=12,
                dynamic=True,
            )

            self.model.export(
                format="torchscript",
                imgsz=640,
                optimize=True,
            )

            if self.device != "cpu":
                self.model.export(
                    format="engine",
                    imgsz=640,
                    device=0,
                )
            
        except Exception as e:
            print(f"Ошибка при экспорте: {e}")
    
    def test_inference(self):
        try:
            test_images = []

            for split in ["train", "val"]:
                img_dir = os.path.join(self.project_dir, "data", split, "images")
                if os.path.exists(img_dir):
                    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))][:2]
                    test_images.extend([os.path.join(img_dir, img) for img in images])
            
            if not test_images:
                return
            
            for i, img_path in enumerate(test_images):
                if os.path.exists(img_path):                
                    results = self.model.predict(
                        source=img_path,
                        conf=0.25,
                        iou=0.45,
                        device=self.device,
                        save=True,
                        save_txt=True,
                        save_conf=True,
                        project=os.path.join(self.project_dir, "runs/predict"),
                        name=f"test_{i+1}",
                        exist_ok=True,
                        show_labels=True,
                        show_conf=True,
                    )
                    
                    if results and len(results) > 0:
                        r = results[0]
    
                        if len(r.boxes) > 0:
                            for j, box in enumerate(r.boxes[:3]):
                                cls_name = self.model.names[int(box.cls)] if hasattr(self.model, 'names') else f"class_{int(box.cls)}"
                                print(f"      {j+1}. {cls_name}: уверенность {box.conf:.3f}")

            
        except Exception as e:
            print(f"Ошибка при тестировании: {e}")
    
    def run(self):
        if not self.setup():
            return

        train_results = self.train()
        if train_results is None:
            return
        
        self.validate()
        self.export_model()
        self.test_inference()


def main():
    trainer = YOLOTrainer()
    trainer.run()

if __name__ == "__main__":
    main()