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
【例10-1】使用k均值聚类算法对所给图像进行颜色分割，并图像分割成5类，最后直观的展示结果。
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys


# 读取图像
image = cv2.imread('image/Example-Bridge.jpg')
if image is None:
    print('没有找到图像')
    sys.exit()

h, w, s = image.shape

# 使用纯白、三种灰色和纯黑进行测试
colors = [(255, 255, 255), (192, 192, 192), (128, 128, 128), (64, 64, 64), (0, 0, 0)]

# 构建图像数据
data = image.reshape((-1, 3))
data = np.float32(data)

# 定义迭代算法终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置图像分割的类别
num_clusters = 5

# 图像分割
ret, labels, centers = cv2.kmeans(data, num_clusters, None, criteria, num_clusters, cv2.KMEANS_RANDOM_CENTERS)

# 根据定义的颜色集合对不同类别的图像区域进行填充
result_data = np.array([colors[int(label)] for label in labels.flatten()])

# 将数据转换为 uint8 类型
result_data = np.clip(result_data, 0, 255).astype(np.uint8)

# 还原图像尺寸
result = result_data.reshape((h, w, s))

# 显示图片
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=0.05, hspace=0)

# 显示原始图片
plt.subplot(1, 2, 1)  
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
plt.title('Origin')  
plt.axis('off')  

# 显示处理后的图片  
plt.subplot(1, 2, 2)  
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  
plt.title('Result')  
plt.axis('off')  

plt.tight_layout()  
plt.show()