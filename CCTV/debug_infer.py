from ultralytics import YOLO
import os

candidates = [
    'runs/detect/mask_finetune/weights/best.pt',
]
model_path = 'yolov8n.pt'
for p in candidates:
    if os.path.exists(p):
        model_path = p
        break

print('Using model:', model_path)
model = YOLO(model_path)
print('Model names:', model.names)

img = 'dataset/mask_dataset/images/val/0001.jpg'
if not os.path.exists(img):
    # try other sample
    imgs = []
    for root, dirs, files in os.walk('dataset'):
        for f in files:
            if f.lower().endswith('.jpg') or f.lower().endswith('.png'):
                imgs.append(os.path.join(root,f))
    if imgs:
        img = imgs[0]
    else:
        raise SystemExit('No image found to test')

print('Testing image:', img)
res = model(img, imgsz=416)
for r in res:
    try:
        boxes = list(r.boxes)
        print('raw boxes count:', len(boxes))
        for b in boxes:
            print('box:', int(b.cls[0]), float(b.conf[0]), list(map(int, b.xyxy[0])))
    except Exception as e:
        print('No boxes or error:', e)

print('Done')
