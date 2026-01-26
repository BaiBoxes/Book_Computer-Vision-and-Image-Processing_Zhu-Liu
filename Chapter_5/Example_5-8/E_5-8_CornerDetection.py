# -*- coding: utf-8 -*-
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
【例5-8】在500*500像素的空白纸上，生成三条相互相交的直线，并以空白纸左上角作为原点，计算交点的坐标，作为真实角点位置，然后分别用Harris角点检测、Shi-Tomasi角点检测并对Shi-Tomasi角点进行亚像素化，然后绘制图像上的角点，最后比对各方法的角点数量，与真实角点的误差以此来比较三种角点检测方法。

"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 创建一个白色背景图像
image_size = 500
image = np.ones((image_size, image_size), dtype=np.uint8) * 255

# 定义直线的端点
linePoint = [[(0, 0), (500, 500)], [(500, 0), (0, 500)], [(100, 0), (100, 500)], ]

# 在图像上绘制这些直线
cv2.line(image, linePoint[0][0], linePoint[0][1], (0, 0, 0), 2)
cv2.line(image, linePoint[1][0], linePoint[1][1], (0, 0, 0), 2)
cv2.line(image, linePoint[2][0], linePoint[2][1], (0, 0, 0), 2)

# 计算直线的交点
def find_intersection(p1, p2):
    A1, B1 = p1[0][1] - p1[1][1], p1[1][0] - p1[0][0]
    C1 = A1 * p1[0][0] + B1 * p1[0][1]
    
    A2, B2 = p2[0][1] - p2[1][1], p2[1][0] - p2[0][0]
    C2 = A2 * p2[0][0] + B2 * p2[0][1]
    
    det = A1 * B2 - A2 * B1
    if det == 0:
        return None
    else:
        x = (C1 * B2 - C2 * B1) / det
        y = (A1 * C2 - A2 * C1) / det
        return x, y

# 找到所有的交点
intersection1 = find_intersection(linePoint[0], linePoint[1])
intersection2 = find_intersection(linePoint[0], linePoint[2])
intersection3 = find_intersection(linePoint[1], linePoint[2])

# 绘制交点
image_intersections = image.copy()
if intersection1:
    cv2.circle(image_intersections, (int(intersection1[0]), int(intersection1[1])), 10, (160, 0, 0), -1)
if intersection2:
    cv2.circle(image_intersections, (int(intersection2[0]), int(intersection2[1])), 10, (160, 0, 0), -1)
if intersection3:
    cv2.circle(image_intersections, (int(intersection3[0]), int(intersection3[1])), 10, (160, 0, 0), -1)

# Harris角点检测
harris_dst = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.01)
harris_dst = cv2.normalize(harris_dst, None, 0, 255, cv2.NORM_MINMAX)
threshold = 0.5 * harris_dst.max()
harris_corners = np.argwhere(harris_dst > threshold)

# 在原始图像上绘制 Harris 角点
harris_image = image.copy()
for corner in harris_corners:
    cv2.circle(harris_image, tuple(corner[::-1]), 10, (160, 0, 0), -1)

# Shi-Tomasi角点检测
shi_tomasi_corners = cv2.goodFeaturesToTrack(image, maxCorners=30, qualityLevel=0.02, minDistance=20)
shi_tomasi_corners = np.float32(shi_tomasi_corners)

# 在原始图像上绘制 Shi-Tomasi 角点
shi_tomasi_image = image.copy()
for corner in shi_tomasi_corners:
    x, y = corner.ravel()
    x, y = int(x), int(y)
    cv2.circle(shi_tomasi_image, (x, y), 10, (160, 0, 0), -1)

# 亚像素级别角点检测
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.02)
subpix_corners = cv2.cornerSubPix(image, shi_tomasi_corners, (5, 5), (-1, -1), criteria)

# 在原始图像上绘制亚像素级别角点
subpix_image = image.copy()
for corner in subpix_corners:
    x, y = corner.ravel()
    x, y = int(x), int(y)
    cv2.circle(subpix_image, (x, y), 10, (160, 0, 0), -1)

# 输出真实角点与各方法检测的角点
print(f'真实角点：{intersection1} {intersection2} {intersection3}')
print(f'Harris角点检测：{harris_corners}')
print(f'Shi-Tomasi角点检测：{shi_tomasi_corners}')
print(f'基于Shi-Tomasi的亚像素角点检测：{subpix_corners}')

# 显示图像
fig, axes = plt.subplots(2, 2, figsize=(6, 6))
plt.subplots_adjust(wspace=0.05, hspace=0.2)
axes[0, 0].imshow(cv2.cvtColor(image_intersections, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Original Image with Lines')
axes[0, 1].imshow(cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Harris Corner Detection')
axes[1, 0].imshow(cv2.cvtColor(shi_tomasi_image, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Shi-Tomasi Corner Detection')
axes[1, 1].imshow(cv2.cvtColor(subpix_image, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Subpixel Corner Detection')

for ax in axes.flat:
    ax.axis('off')

plt.show()