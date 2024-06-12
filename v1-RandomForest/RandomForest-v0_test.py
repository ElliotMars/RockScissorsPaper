import os
import cv2
import joblib
import numpy as np
from dataloader import load_images_from_folder  # 确保导入正确
from imageprocess import Pre_process_img, feature_extraction_img

# 加载模型
model_path = 'models/random_forest_model_20240611_191026.joblib'
clf = joblib.load(model_path)
print(f"Model loaded from {model_path}")
category = ['Paper', 'Rock', 'Scissors']
# 加载新的图片并进行预测
def predict_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (300, 300))  # 确保图像尺寸为300x300
        # 特征提取
        result = Pre_process_img(img)
        feature = np.array(feature_extraction_img(result)).reshape(1, -1)
        print(feature)
        prediction = clf.predict(feature)
        return prediction[0]
    else:
        return "Image not found"

# 示例
new_image_path = '../myhandtest/Rock3.png'  # 修改为你的新图像路径
prediction = predict_image(new_image_path)
class_result = category[prediction]
print(f"Prediction for the new image: {class_result}")
