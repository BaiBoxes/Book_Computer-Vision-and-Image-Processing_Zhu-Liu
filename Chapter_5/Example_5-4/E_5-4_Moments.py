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
【例5-4】使用cv2.moments函数计算图像中每个轮廓的空间矩、中心矩和归一化中心矩，并计算质心。接着使用cv2.HuMoments函数计算得到图像的Hu矩，并显示所有结果。
"""


import cv2
import numpy as np


# 读取图像
image = cv2.imread('image/Example-Vibration.jpg')

# 创建红色掩码
mask = cv2.inRange(image, np.array([0, 0, 180]), np.array([100, 100, 255]))

# 只保留振动测试图红色区域
red_only = cv2.bitwise_and(image, image, mask=mask)

# 转换为灰度图
gray = cv2.cvtColor(red_only, cv2.COLOR_BGR2GRAY)

# 查找图中的轮廓
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 过滤面积和周长的轮廓
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    length = cv2.arcLength(contour, True)
    if area > 150 and length > 10:
        filtered_contours.append(contour)
# 计算图像矩
output_image = image.copy()
for i, contour in enumerate(filtered_contours):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 计算质心
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        # 绘制质心与轮廓
        cv2.circle(output_image, (cx, cy), 5, (0, 0, 255), -1) 
        cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 3) 

    # 输出空间矩、中心矩、归一化中心矩和Hu矩
    print(f"轮廓 {i + 1}:")

    print(f"[空间矩] m00: {moments['m00']:.4g}, m10: {moments['m10']:.4g}, m01: {moments['m01']:.4g}")

    if moments['m00'] != 0:
        print(f"质心坐标: ({cx}, {cy})")

    print(f"[中心矩] mu20: {moments['mu20']:.4g}, mu11: {moments['mu11']:.4g}, mu02: {moments['mu02']:.4g}")

    print(f"[归一化中心矩] nu20: {moments['nu20']:.4g}, nu11: {moments['nu11']:.4g}, nu02: {moments['nu02']:.4g}")

    print("[Hu矩]:")
    for j in range(7):
        print(f"Hu[{j + 1}]: {hu_moments[j]:.4g}")
    print()

# 显示质心和轮廓的图像
cv2.imshow('Centroids and Contours', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()