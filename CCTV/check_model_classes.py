"""
Quick script to check what classes are in your best.pt model
"""
from ultralytics import YOLO

print("Loading best.pt model...")
model = YOLO("best.pt")

print("\n" + "="*60)
print("MODEL CLASSES:")
print("="*60)
for class_id, class_name in model.names.items():
    print(f"  {class_id}: {class_name}")
print("="*60)

print("\nTotal classes:", len(model.names))
