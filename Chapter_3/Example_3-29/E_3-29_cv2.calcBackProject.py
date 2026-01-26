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
【例3-29】计算目标图像的直方图模型，然后将待处理的图像用cv2.calcBackProject函数计算出反向投影图。
        最后显示目标图像、待处理图像与相应的直方图以及计算出的反向投影图。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取目标图像和待处理图像
target_image = cv2.imread('image/backproject_target.jpg', cv2.COLOR_BGR2RGB)
input_image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.COLOR_BGR2RGB)
# 计算目标图像的直方图模型
target_hist = cv2.calcHist([target_image], [0], None, [256], [0, 256])
target_hist = cv2.normalize(target_hist, None, alpha=1, beta=0, norm_type=cv2.NORM_INF)
input_hist = cv2.calcHist([input_image], [0], None, [256], [0, 256])
input_hist = cv2.normalize(input_hist, None, alpha=1, beta=0, norm_type=cv2.NORM_INF)
# 使用cv2.calcBackProject计算反向投影图
backproject = cv2.calcBackProject([input_image], [0], target_hist, [0, 256], 1)
# 显示目标图像、待处理图像和反向投影图等
plt.figure(num='直方图反向投影', figsize=(12, 8))
# 目标图像
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
plt.title('Target Image')
plt.axis('off')
# 待处理图像
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')
# 反向投影图
plt.subplot(2, 3, 3)
plt.imshow(backproject, cmap='gray')
plt.title('Backproject')
plt.axis('off')
# 目标图像的直方图
plt.subplot(2, 3, 4)
plt.bar(np.arange(256), target_hist.reshape(256), width=1.0)
plt.title('Target Histogram')
plt.xlabel('Gray Value')
plt.ylabel('Frequency')
# 待处理图像的直方图
plt.subplot(2, 3, 5)
plt.bar(np.arange(256), input_hist.reshape(256), width=1.0)
plt.title('Input Histogram')
plt.xlabel('Gray Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
