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
【例5-2】根据所给图像中的轮廓，选择适当的轮廓检索模式和逼近方法，使用cv2.drawContours函数检查轮廓，然后利用cv2.drawContours函数绘制物体的外围和内部轮廓，区分不同轮廓层次。计算每个轮廓的面积和周长以分析轮廓的几何特征，最后显示出输入图像、轮廓检测结果与每个轮廓的面积和周长。
"""


import cv2
import matplotlib.pyplot as plt


# 读取彩色图像
image = cv2.imread('image/Example-Bridge.jpg')

# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用自适应阈值进行二值化
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# 检测轮廓
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 过滤面积和周长大于0的轮廓
filtered_contours = []
filtered_hierarchy = []
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    length = cv2.arcLength(contour, True)
    if area > 50 and length > 40:
        filtered_contours.append(contour)
        filtered_hierarchy.append(hierarchy[0][i])

# 创建一个副本用于绘制轮廓
contour_image = image.copy()

# 绘制筛选后的轮廓
for i, contour in enumerate(filtered_contours):
    # 区分外部轮廓和内部轮廓
    if filtered_hierarchy[i][3] == -1:
        # 绿色，外部轮廓
        color = (0, 255, 0)
    else:
        # 红色，内部轮廓
        color = (0, 0, 255)

    thickness = 1
    cv2.drawContours(contour_image, filtered_contours, i, color, thickness, cv2.LINE_8)

# 计算并输出筛选后的轮廓的面积和周长
for i, contour in enumerate(filtered_contours):
    area = cv2.contourArea(contour)
    length = cv2.arcLength(contour, True)
    print(f"轮廓 {i} 的面积: {area:.2f}, 周长: {length:.2f}")

# 显示输入图像、二值化图像和轮廓检测结果
plt.figure(figsize=(15, 5))

# 显示原始图像
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# 显示二值化图像
plt.subplot(1, 3, 2)
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# 显示轮廓检测结果
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title('Filtered Contours')
plt.axis('off')

plt.show()