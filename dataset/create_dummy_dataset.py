"""
Create a tiny dummy YOLO-format dataset for quick smoke tests.

This will create:
 - dataset/mask_dataset/images/train/0001.jpg
 - dataset/mask_dataset/labels/train/0001.txt
 - dataset/mask_dataset/images/val/0001.jpg
 - dataset/mask_dataset/labels/val/0001.txt

Each label contains one box (class 0) centered in the image.

Run:
  python dataset/create_dummy_dataset.py
"""
import os
import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "mask_dataset")

def ensure_dirs():
    for split in ("train", "val"):
        img_dir = os.path.join(ROOT, "images", split)
        lbl_dir = os.path.join(ROOT, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

def create_image(path, w=320, h=240, color=(200,200,200)):
    img = np.full((h,w,3), color, dtype=np.uint8)
    # draw a simple person-like rectangle
    cv2.rectangle(img, (int(w*0.4), int(h*0.2)), (int(w*0.6), int(h*0.8)), (50,50,50), -1)
    cv2.imwrite(path, img)

def create_label(path):
    # one box covering the rectangle drawn above; YOLO normalized: class x_center y_center w h
    x_center = 0.5
    y_center = 0.5
    box_w = 0.2
    box_h = 0.6
    with open(path, "w") as f:
        f.write(f"0 {x_center} {y_center} {box_w} {box_h}\n")

def main():
    ensure_dirs()
    for split in ("train", "val"):
        img_path = os.path.join(ROOT, "images", split, "0001.jpg")
        lbl_path = os.path.join(ROOT, "labels", split, "0001.txt")
        create_image(img_path)
        create_label(lbl_path)

    print("Created dummy dataset at:", ROOT)

if __name__ == "__main__":
    main()
