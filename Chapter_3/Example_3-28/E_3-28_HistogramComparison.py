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
【例3-28】将图像与上下翻转后的图像和另一张完全不同的图像用cv2.compareHist函数分别进行直方图比较，然后显示比较结果，并将原图与对应直方图显示出来。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_COLOR)
# 将图像上下翻转
flipped_image = cv2.flip(image, 0)
# 读取另一张完全不同的图像
different_image = cv2.imread('image/Example-BridgeBottom_gray.jpg', cv2.IMREAD_COLOR)
# 计算图像的直方图
hist_image = cv2.calcHist([image], [0], None, [256], [0, 255])
hist_image = cv2.normalize(hist_image, None, alpha=1, beta=0, norm_type=cv2.NORM_INF)
hist_flipped = cv2.calcHist([flipped_image], [0], None, [256], [0, 255])
hist_flipped = cv2.normalize(hist_flipped, None, alpha=1, beta=0, norm_type=cv2.NORM_INF)
hist_different = cv2.calcHist([different_image], [0], None, [256], [0, 255])
hist_different = cv2.normalize(hist_different, None, alpha=1, beta=0, norm_type=cv2.NORM_INF)
# 比较图像与上下翻转后的图像的直方图
result_flip = cv2.compareHist(hist_image, hist_flipped, cv2.HISTCMP_CORREL)
# 比较图像与完全不同的图像的直方图
result_diff = cv2.compareHist(hist_image, hist_different, cv2.HISTCMP_CORREL)
# 显示原图与上下翻转后的图像以及直方图比较结果
plt.figure(num='直方图比较', figsize=(12, 6))
# 原图
plt.subplot(241)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
# 上下翻转后的图像
plt.subplot(242)
plt.imshow(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
plt.title('Flipped Image')
plt.axis('off')
# 完全不同的图像
plt.subplot(243)
plt.imshow(cv2.cvtColor(different_image, cv2.COLOR_BGR2RGB))
plt.title('Different_Image')
plt.axis('off')
# 原图的直方图
plt.subplot(245)
plt.bar(np.arange(256), hist_image.reshape(256), width=1.0)
plt.title('Histogram (Original Image)')
plt.xlabel('Hue-Saturation Value')
plt.ylabel('Frequency')
# 上下翻转后的图像的直方图
plt.subplot(246)
plt.bar(np.arange(256), hist_flipped.reshape(256), width=1.0)
plt.title('Histogram (Flipped Image)')
plt.xlabel('Hue-Saturation Value')
plt.ylabel('Frequency')
# 不同图像的直方图
plt.subplot(247)
plt.bar(np.arange(256), hist_different.reshape(256), width=1.0)
plt.title('Histogram (Different Image)')
plt.xlabel('Hue-Saturation Value')
plt.ylabel('Frequency')
# 直方图比较结果
plt.subplot(248)
bar_scores = [result_flip, result_diff]
print(bar_scores)
bars = plt.bar(['Original vs Flipped', 'Original vs Different'], bar_scores)
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{bar_scores[i]:.2f}', ha='center')
plt.title('Histogram Comparison Result')
plt.ylabel('Correlation Score')
plt.tight_layout()
plt.show()