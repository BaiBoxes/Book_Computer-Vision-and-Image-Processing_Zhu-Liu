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
【例3-34】在一幅图上产生椒盐噪声后，分别用cv2.blur函数进行均值滤波、cv2.medianBlur函数进行中值滤波
            与cv2.bilateralFilter函数进行双边滤波，然后显示出滤波结果并分析。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 生成一幅带有椒盐噪声的图像
def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """ 添加椒盐噪音的函数 """
    noisy_image = image.copy()
    total_pixels = image.size
    # 添加椒盐噪音
    num_salt = np.ceil(total_pixels * salt_prob)
    salt_coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    # 添加胡椒噪音
    num_pepper = np.ceil(total_pixels * pepper_prob)
    pepper_coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    return noisy_image

# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 添加椒盐噪声
salt_and_pepper_image = add_salt_and_pepper_noise(image)
# 均值滤波
blur_image = cv2.blur(salt_and_pepper_image, (3, 3))
# 中值滤波
median_filtered_image = cv2.medianBlur(salt_and_pepper_image, 3)
# 双边滤波
bilateral_filtered_image = cv2.bilateralFilter(salt_and_pepper_image, d=9, sigmaColor=75, sigmaSpace=75)
# 显示图像及滤波结果
plt.figure(num='Filtered Image', figsize=(9, 9))
plt.subplot(221)
plt.imshow(cv2.cvtColor(salt_and_pepper_image, cv2.COLOR_BGR2RGB))
plt.title('Noisy Image')
plt.subplot(222)
plt.imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
plt.title('Mean Filtered Image')
plt.subplot(223)
plt.imshow(cv2.cvtColor(median_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Median Filtered Image')
plt.subplot(224)
plt.imshow(cv2.cvtColor(bilateral_filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Bilateral Filtered Image')
plt.show()