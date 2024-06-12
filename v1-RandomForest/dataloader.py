import os
import cv2

def load_images_from_folder(folder):
    images = []
    labels = []
    target_size = (300, 300)  # 设置目标尺寸

    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(label_folder, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        # 检查图像尺寸，如果不是目标尺寸则调整大小
                        if img.shape[:2] != target_size:
                            img = cv2.resize(img, target_size)
                        images.append(img)
                        labels.append(label)
    return images, labels
