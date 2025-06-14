import os
import shutil
import random

def split_dataset(input_dir, output_dir):
    split_ratio=0.8
    classes = os.listdir(input_dir)
    #check all plant classes to train model
    for cls in classes:
        class_dir = os.path.join(input_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        images = os.listdir(class_dir)
        random.shuffle(images)
        
        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        train_cls_dir = os.path.join(output_dir, "train", cls)
        val_cls_dir = os.path.join(output_dir, "val", cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir, exist_ok=True)
        
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_cls_dir, img))

        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_cls_dir, img))

if __name__ == "__main__":
    input_dir = "data/PlantVillage"
    output_dir = "data"
    split_dataset(input_dir,(output_dir))