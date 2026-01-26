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
【例5-5】在500*500像素的空白纸上随机生成两种离散点集，在分别是使用cv2.fitLine函数和cv2.fitEllipse函数各自对其中一种离散点集进行拟合，并分
别显示拟合结果。
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 创建一个500x500的空白图像
image_line = np.ones((500, 500, 3), dtype=np.uint8) * 255
image_ellipse = np.ones((500, 500, 3), dtype=np.uint8) * 255


# 生成第一种离散点集（适合直线拟合）
num_points_line = 50
line_x = np.linspace(50, 500 - 50, num_points_line)
line_y = 0.5 * line_x + 100

# 添加噪声以使点围绕直线随机分布
noise = np.random.normal(0, 20, num_points_line)
points_line = np.vstack((line_x, line_y + noise)).T

# 生成第二种离散点集（适合椭圆拟合）
num_points_ellipse = 50
angles_ellipse = np.linspace(0, 2 * np.pi, num_points_ellipse)
radii_ellipse_x = np.random.uniform(0, 200, num_points_ellipse)
radii_ellipse_y = np.random.uniform(0, 130, num_points_ellipse)
x_points_ellipse = (500 // 2 + radii_ellipse_x * np.cos(angles_ellipse)).astype(np.float32)
y_points_ellipse = (500 // 2 + radii_ellipse_y * np.sin(angles_ellipse)).astype(np.float32)
points_ellipse = np.vstack((x_points_ellipse, y_points_ellipse)).T

# 绘制离散点集
for pt in points_line:
    cv2.circle(image_line, tuple(pt.astype(int)), 6, (0, 0, 0), -1)

for pt in points_ellipse:
    cv2.circle(image_ellipse, tuple(pt.astype(int)), 6, (0, 0, 0), -1)

# 使用cv2.fitLine对第一组点集进行线性拟合
[vx, vy, x0, y0] = cv2.fitLine(points_line, cv2.DIST_L2, 0, 0.01, 0.01)
left_y = int((-x0 * vy / vx) + y0)
right_y = int(((image_line.shape[1] - x0) * vy / vx) + y0)
cv2.line(image_line, (image_line.shape[1] - 1, right_y), (0, left_y), (255, 0, 0), 2)

# 使用cv2.fitEllipse对第二组点集进行椭圆拟合
if len(points_ellipse) >= 5:
    ellipse = cv2.fitEllipse(points_ellipse)
    cv2.ellipse(image_ellipse, ellipse, (0, 255, 0), 2)


# 显示结果
plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.05, hspace=0)

# 显示直线拟合结果
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_line, cv2.COLOR_BGR2RGB))
plt.title('Line Fitting')
plt.axis('off')

# 显示椭圆拟合结果
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_ellipse, cv2.COLOR_BGR2RGB))
plt.title('Ellipse Fitting')
plt.axis('off')

plt.show()