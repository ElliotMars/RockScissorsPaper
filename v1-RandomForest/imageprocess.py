import cv2

def binarize_images(images):
    binarized_images = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        binarized_images.append(binary)
    return binarized_images

def extract_contours(binarized_images):
    contours_list = []
    for binary in binarized_images:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.append(contours[0])  # 假设只有一个主要轮廓
    return contours_list


def calculate_features(contours_list):
    features = []
    for contour in contours_list:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        # 处理 hull_area 为零的情况
        if hull_area == 0:
            solidity = 0  # 或者使用其他合适的默认值
        else:
            solidity = float(area) / hull_area

        rect_area = w * h
        extent = float(area) / rect_area
        perimeter = cv2.arcLength(contour, True)
        feature = [aspect_ratio, area, solidity, extent, perimeter]
        features.append(feature)
    return features

def Pre_process(img):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    # 图像膨胀
    dilate = cv2.dilate(img, element)
    # 图像腐蚀
    erode = cv2.erode(img, element)
    # 获取图像的边缘
    result = cv2.absdiff(dilate, erode)
    # 二值化图像
    retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)
    # 反色处理
    result = cv2.bitwise_not(result)
    # 中值滤波去除噪声
    result = cv2.medianBlur(result, 23)
    return result
