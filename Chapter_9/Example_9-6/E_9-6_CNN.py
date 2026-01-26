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
【例9-6】使用TensorFlow库构建一个CNN模型，并对所给数据集进行训练和验证，以便对新图像进行分类预测。
"""

import numpy as np
import cv2
import os
from tensorflow import keras
import tensorflow as tf
layers = tf.keras.layers
from sklearn.model_selection import train_test_split

# 自定义加载图像的函数
def load_images(folder):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # 将图像调整为 100x100
            img_resized = cv2.resize(img, (100, 100))
            images.append(img_resized)
            filenames.append(filename)

            # 从文件名中提取标签
            label = filename.split('_')[0]
            if label == 'circle':
                labels.append(0)
            elif label == 'square':
                labels.append(1)
            elif label == 'star':
                labels.append(2)
            elif label == 'triangle':
                labels.append(3)

    return np.array(images, dtype='float32'), np.array(labels, dtype='int32'), filenames

# 训练模型部分
train_images, train_labels, _ = load_images('image\\Learn\\train_image')

# 数据预处理
train_images = train_images / 255.0  # 归一化到 [0, 1]
train_images = train_images.reshape(-1, 100, 100, 1)  # 重塑为 (样本数, 高度, 宽度, 通道数)

# 划分训练和验证集
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# 构建CNN模型
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4个类别
])

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# 保存模型
model.save('./results/cnn_model.h5')
print('模型训练完成并已保存。')


# 加载模型
model = keras.models.load_model('./results/cnn_model.h5')

# 加载预测数据
test_images, test_labels, filenames = load_images('image\\Learn\\predict_image')
test_images = test_images / 255.0  # 归一化到 [0, 1]
test_images = test_images.reshape(-1, 100, 100, 1)  # 重塑为 (样本数, 高度, 宽度, 通道数)

# 进行预测
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# 记录每种类别的正确预测和总预测数
category_stats = {
    0: {'correct': 0, 'total': 0},  # circle
    1: {'correct': 0, 'total': 0},  # square
    2: {'correct': 0, 'total': 0},  # star
    3: {'correct': 0, 'total': 0}   # triangle
}

correct_count = 0
for i in range(len(predicted_labels)):
    predicted_label = predicted_labels[i]
    actual_label = int(test_labels[i])

    # 更新每种类别的统计信息
    category_stats[actual_label]['total'] += 1
    if predicted_label == actual_label:
        category_stats[actual_label]['correct'] += 1
        correct_count += 1

    print('图像 {} 的预测类别为：{}，实际类别为：{}'.format(filenames[i], predicted_label, actual_label))

accuracy = correct_count / len(predicted_labels)
print(f'预测完成。整体准确率为：{accuracy:.2%}')

# 输出每种类别的准确率
for label, stats in category_stats.items():
    if stats['total'] > 0:
        class_accuracy = stats['correct'] / stats['total']
        print(f'类别 {label} 的准确率为：{class_accuracy:.2%}')
    else:
        print(f'类别 {label} 的准确率无法计算，因为没有样本。')