import os, random, shutil

src = "mask_dataset/images/train"
val = "mask_dataset/images/val"

images = os.listdir(src)
random.shuffle(images)

split = int(len(images) * 0.2)

for img in images[:split]:
    shutil.move(
        os.path.join(src, img),
        os.path.join(val, img)
    )

print("Split done")
