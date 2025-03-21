from dataloader import load_images_from_folder
from imageprocess import Pre_process, feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
from datetime import datetime

#加载数据
images, labels = load_images_from_folder('../dataset/data')

#特征提取
results = Pre_process(images)
features = feature_extraction(results)

#训练
X = features
y = labels

# 将标签转换为数值型
le = LabelEncoder()
y = le.fit_transform(y)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
report = classification_report(y_test, y_pred, target_names=le.classes_)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(report)

# 获取当前时间戳
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 保存模型
model_path = f'models/random_forest_model_{timestamp}.joblib'
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# 保存评估结果为 TXT 文件
report_txt_path = f'models/model_evaluation_{timestamp}.txt'
with open(report_txt_path, 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(report)
print(f"Evaluation report saved to {report_txt_path}")