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
【例7-2】使用cv2.resize函数双三次插值法对图片进行超分辨率成像以提高图片的分辨率，并使用边缘增强滤波器增强图像细节。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt 

# 打开图片
input_image = cv2.imread('image/Example-Bridge.jpg')

# 使用双三次插值法进行超分辨率成像  
resized_image = cv2.resize(input_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)  

# 使用边缘增强滤波器来提高清晰度  
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  
enhanced_image = cv2.filter2D(resized_image, -1, kernel)  

# 显示图片
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=0.05, hspace=0)

# 显示原始图片
plt.subplot(1, 2, 1)  
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))  
plt.title('Original Image\nResolution: {}x{}'.format(input_image.shape[1], input_image.shape[0]))  
plt.axis('off')  

# 显示处理后的图片  
plt.subplot(1, 2, 2)  
plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))  
plt.title('Super-Resolution Image\nResolution: {}x{}'.format(enhanced_image.shape[1], enhanced_image.shape[0]))  
plt.axis('off')  

plt.tight_layout()  
plt.show()











