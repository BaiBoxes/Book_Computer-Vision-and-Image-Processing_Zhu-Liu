# -*- coding: utf-8 -*-
"""
------------------------
版权声明：
本书内部包含的代码示例、算法和技术解释是受到知识产权法律保护的。
这些代码示例和相关内容仅用于学习和教育目的，以帮助读者更好地理解书中的概念和知识。
版权归属于书籍的作者或权利人所有。
    这些代码示例和技术解释的使用受到以下限制：
        代码示例仅用于学习和教育用途。读者可以结合书籍内容，在非商业的环境中使用这些代码示例，进行学习、实验和练习。
        代码示例不得用于商业用途，包括但不限于出售、分发以获取利润、嵌入商业软件或产品中。
        如需使用、修改和分发代码示例，其根据GNU Affero通用公共许可证(AGPL)3.0版本授权，请遵循源代码可用性与网络互动条款。
        任何对代码示例的修改、衍生或重新分发，应该在适当的情况下保留原作者的权利声明，并在代码中进行明确标注。
        代码示例和技术解释的使用不得侵犯任何第三方的知识产权，包括但不限于专利、商标、版权等。
        本书作者和出版社对于读者因使用这些代码示例导致的任何损失或风险概不负责。
请尊重知识产权，遵守以上声明，合理使用本书中的代码示例和相关内容。
------------------------
版权归属于：清华大学出版社 and 《计算机视觉与图像处理》作者
------------------------
【例5-3】对斜拉桥图进行形状检测，用直线检测识别索结构，用多边形检测识别桥板结构，并用凸包检测降低轮廓复杂度。对振动测试图中的红色标志进行圆检查，然后对两个红色标志进行形状匹配。最后显示结果。
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取桥梁图与振动测试图
image_b = cv2.imread('image/Example-Bridge.jpg')
image_v = cv2.imread('image/Example-Vibration.jpg')

# 创建红色掩码
mask = cv2.inRange(image_v, np.array([0, 0, 180]), np.array([100, 100, 255]))
# 只保留振动测试图红色区域
red_only = cv2.bitwise_and(image_v, image_v, mask=mask)

# 转换为灰度图
gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
gray_v = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)

# 应用高斯模糊以减少噪声
blurred_b = cv2.GaussianBlur(gray_b, (5, 5), 0)
blurred_v = cv2.GaussianBlur(gray_v, (5, 5), 0)

# 检测图像中的边缘
edges_b = cv2.Canny(blurred_b, 20, 150)
edges_v = cv2.Canny(blurred_v, 20, 150)

# 复制图像用于绘制
image_b_lines = image_b.copy()
image_b_polygons = image_b.copy()
image_v_circles = image_v.copy()

# 使用霍夫变换检测直线
lines = cv2.HoughLinesP(edges_b, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_b_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 查找桥梁轮廓
contours_b, _ = cv2.findContours(edges_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours_b:
    area = cv2.contourArea(contour)
    if area > 150:
        # 使用多边形逼近桥板结构
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image_b_polygons, [approx], 0, (0, 255, 0), 2)

        # 计算并绘制凸包，降低轮廓复杂度
        hull = cv2.convexHull(contour)
        cv2.drawContours(image_b_polygons, [hull], 0, (255, 0, 0), 2)

# 查找振动测试图中的轮廓
contours_v, _ = cv2.findContours(edges_v, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_features = []
for contour in contours_v:
    # 过滤掉面积较小的轮廓
    area = cv2.contourArea(contour)
    if area > 150:
        
        # 检测并绘制圆形
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image_v_circles, center, radius, (0, 255, 0), 2)

        # 将符合条件的轮廓添加到特征列表中
        contour_features.append(contour)

# 对检测到的轮廓进行比较
for i in range(len(contour_features)):
    for j in range(i + 1, len(contour_features)):
        similarity = cv2.matchShapes(contour_features[i], contour_features[j], cv2.CONTOURS_MATCH_I1, 0.0)
        print(f'Shape similarity between contour {i} and contour {j}: {similarity:.4f}')


# 显示输入图像和检测结果
plt.figure(figsize=(13, 6))
plt.subplots_adjust(wspace=0.05, hspace=0)

# 显示桥梁原始图像
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
plt.title('Bridge Original Image')
plt.axis('off')

# 显示桥梁边缘检测图像
plt.subplot(2, 4, 2)
plt.imshow(edges_b, cmap='gray')
plt.title('Bridge Edge Detection')
plt.axis('off')

# 显示桥梁直线检测结果
plt.subplot(2, 4, 3)
plt.imshow(cv2.cvtColor(image_b_lines, cv2.COLOR_BGR2RGB))
plt.title('Bridge Line Detection')
plt.axis('off')

# 显示桥梁多边形检测结果
plt.subplot(2, 4, 4)
plt.imshow(cv2.cvtColor(image_b_polygons, cv2.COLOR_BGR2RGB))
plt.title('Bridge Polygon Detection')
plt.axis('off')

# 显示振动测试原始图像
plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(image_v, cv2.COLOR_BGR2RGB))
plt.title('Vibration Original Image')
plt.axis('off')

# 显示振动测试颜色检测图像
plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(red_only, cv2.COLOR_BGR2RGB))
plt.title('Vibration Color Detection')
plt.axis('off')

# 显示振动测试边缘检测结果
plt.subplot(2, 4, 7)
plt.imshow(edges_v, cmap='gray')
plt.title('Vibration Edge Detection')
plt.axis('off')

# 显示振动测试圆检测结果
plt.subplot(2, 4, 8)
plt.imshow(cv2.cvtColor(image_v_circles, cv2.COLOR_BGR2RGB))
plt.title('Vibration Circle Detection')
plt.axis('off')

plt.show()