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
【例10-5】随机生成包含两类像素的yml文件，并使用支持向量机算法，最后直观的展示分类预测结果。
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt


# 生成正弦曲线和余弦曲线的数据点
x_values = np.linspace(0, 640, 200)
y_values_sine = 50 * np.sin(0.01 * x_values) + 120
y_values_cosine = -50 * np.cos(0.01 * x_values) + 360

# 在两条曲线周围生成两类数据点
data = []
labels = []

for x, y_sin, y_cos in zip(x_values, y_values_sine, y_values_cosine):
    for _ in range(10):  # 每个曲线点周围生成10个随机点
        # 生成正弦曲线附近的点
        data.append([np.random.normal(x, 15), np.random.normal(y_sin, 15)])
        labels.append(0)
        # 生成余弦曲线附近的点
        data.append([np.random.normal(x, 15), np.random.normal(y_cos, 15)])
        labels.append(1)

data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# 创建空白图像用于显示点
img = np.ones((480, 640, 3), dtype='uint8') * 255

# 绘制正弦曲线和余弦曲线
for i in range(1, len(x_values)):
    cv2.line(img, (int(x_values[i-1]), int(y_values_sine[i-1])), (int(x_values[i]), int(y_values_sine[i])), (0, 0, 0), 2)
    cv2.line(img, (int(x_values[i-1]), int(y_values_cosine[i-1])), (int(x_values[i]), int(y_values_cosine[i])), (0, 0, 0), 2)

# 绘制原始数据点
for i in range(len(data)):
    x, y = data[i]
    color = (255, 0, 0) if labels[i] == 0 else (0, 0, 0)
    cv2.circle(img, (int(x), int(y)), 3, color, -1)

# 建立SVM模型
svm = cv2.ml.SVM_create()

# 设置SVM参数
svm.setKernel(cv2.ml.SVM_RBF)
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(5.831)
svm.setGamma(0.01)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

# 训练SVM模型
svm.train(data, cv2.ml.ROW_SAMPLE, labels)

# 保存训练好的 SVM 模型到 'svm_model.yml'
svm.save('./Chapter_9/Example_9-5/results/svm_model.yml')
print('模型训练完成并已保存。')

# 用模型对图像中的全部像素进行分类
classification_img = np.ones((480, 640, 3), dtype='uint8') * 255
for i in range(0, classification_img.shape[1]):
    for j in range(0, classification_img.shape[0]):
        sample = np.array([[i, j]], dtype='float32')
        _, res = svm.predict(sample)
        color = (211, 211, 211) if res == 0 else (0, 255, 225)
        classification_img[j, i] = color

# 在分类结果图像中绘制原始数据点
for i in range(len(data)):
    x, y = data[i]
    color = (255, 0, 0) if labels[i] == 0 else (0, 0, 0)
    cv2.circle(classification_img, (int(x), int(y)), 3, color, -1)

# 分别显示原始数据图像和分类结果图像
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=0.05, hspace=0)

# 显示原始图片
plt.subplot(1, 2, 1)  
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
plt.title('Original Data')  
plt.axis('off')  

# 显示处理后的图片  
plt.subplot(1, 2, 2)  
plt.imshow(cv2.cvtColor(classification_img, cv2.COLOR_BGR2RGB))  
plt.title('SVM Classification Result')  
plt.axis('off')  

plt.tight_layout()  
plt.show()