from dataloader import load_images_from_folder
from imageprocess import binarize_images, extract_contours, calculate_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#加载数据
images, labels = load_images_from_folder('dataset/data')

#特征提取
binarized_images = binarize_images(images)
contours_list = extract_contours(binarized_images)
features = calculate_features(contours_list)

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
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

