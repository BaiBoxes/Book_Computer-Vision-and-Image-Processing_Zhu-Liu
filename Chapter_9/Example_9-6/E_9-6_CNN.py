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
model.save('./Chapter_9/Example_9-6/results/cnn_model.h5')
print('模型训练完成并已保存。')


# 加载模型
model = keras.models.load_model('./Chapter_9/Example_9-6/results/cnn_model.h5')

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