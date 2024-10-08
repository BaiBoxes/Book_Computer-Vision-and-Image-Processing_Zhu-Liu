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
【例3-26】将图像用cv2.calcHist进行直方图计算，然后用cv2.normalize函数进行4种归一化，并利用Matplotlib库将原图与直方图显示出来。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 计算直方图
hist = cv2.calcHist([image], [0], None, [256], [0, 255])
# 创建Matplotlib图形窗口
plt.figure(num='归一化直方图', figsize=(12, 6))
# 进行归一化 (NORM_INF)
normalized_hist_inf = cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_INF)
# 进行归一化 (NORM_L1)
normalized_hist_l1 = cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_L1)
# 进行归一化 (NORM_L2)
normalized_hist_l2 = cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_L2)
# 进行归一化 (NORM_MINMAX)
normalized_hist_minmax = cv2.normalize(hist, None, alpha=1, beta=0, norm_type=cv2.NORM_MINMAX)
# 绘图
# 原图
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
# 不进行归一化的直方图
plt.subplot(2, 3, 2)
plt.plot(hist)
plt.title('Histogram (No Normalization)')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
# 进行归一化 (NORM_INF)
plt.subplot(2, 3, 3)
plt.plot(normalized_hist_inf)
plt.title('Normalized Histogram (NORM_INF)')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
# 进行归一化 (NORM_L1)
plt.subplot(2, 3, 4)
plt.plot(normalized_hist_l1)
plt.title('Normalized Histogram (NORM_L1)')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
# 进行归一化 (NORM_L2)
plt.subplot(2, 3, 5)
plt.plot(normalized_hist_l2)
plt.title('Normalized Histogram (NORM_L2)')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
# 进行归一化 (NORM_MINMAX)
plt.subplot(2, 3, 6)
plt.plot(normalized_hist_minmax)
plt.title('Normalized Histogram (NORM_MINMAX)')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.tight_layout()
plt.show()