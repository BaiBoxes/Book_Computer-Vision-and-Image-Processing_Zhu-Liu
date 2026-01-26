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
【例10-3】使用决策树算法构建一个DTrees模型，并对所给数据集进行训练和预测，最后直观的展示分类的准确率和预测结果。
"""


import numpy as np
import cv2
import os


def load_images(folder):
    images = []
    labels = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # 保持图像原始100x100的尺寸
            img_resized = img.reshape((1, 10000))
            images.append(img_resized)
            filenames.append(filename)
            # 从文件名中提取标签（假设文件名为“circle_01.png”、“square_01.png”等）
            label = filename.split('_')[0]
            if label == 'circle':
                labels.append(0)
            elif label == 'square':
                labels.append(1)
            elif label == 'star':
                labels.append(2)
            elif label == 'triangle':
                labels.append(3)

    return np.array(images, dtype='float32'), np.array(labels, dtype='int32')[:, np.newaxis], filenames


# 训练模型部分
train_images, train_labels, _ = load_images('image\\Learn\\train_image')
train_data = train_images.reshape(-1, 10000).astype(np.float32)

# 创建决策树模型
dt = cv2.ml.DTrees_create()

# 设置决策树的最大深度
dt.setMaxDepth(20)
# 设置用于交叉验证的折数
dt.setCVFolds(0)
# 设置每个节点上所需的最小样本数
dt.setMinSampleCount(10)
# 设置是否使用误差标准规则进行树剪枝
dt.setUse1SERule(False)
# 设置是否截断已剪枝的树
dt.setTruncatePrunedTree(False)

# 训练模型
dt.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# 计算训练准确率
retval, results = dt.predict(train_data)
matches = results == train_labels
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / results.size
print('DTrees模型训练的准确率为{}%'.format(accuracy))

# 测试模型对图形的识别情况
test_images, test_labels, filenames = load_images('image\\Learn\\predict_image')
test_data = test_images.reshape(-1, 10000).astype(np.float32)

# 进行图形识别
ret, result = dt.predict(test_data)

# 初始化每个类别的统计信息
category_stats = {0: {'correct': 0, 'total': 0},  # circle
                    1: {'correct': 0, 'total': 0},  # square
                    2: {'correct': 0, 'total': 0},  # star
                    3: {'correct': 0, 'total': 0}}  # triangle

correct_count = 0
for i in range(len(result)):
    predicted_label = int(result[i][0])
    actual_label = int(test_labels[i][0])

    # 更新每种类别的统计信息
    category_stats[actual_label]['total'] += 1
    if predicted_label == actual_label:
        category_stats[actual_label]['correct'] += 1
        correct_count += 1

    print('图像 {} 的预测类别为：{}，实际类别为：{}'.format(filenames[i], predicted_label, actual_label))

# 计算整体准确率
overall_accuracy = correct_count * 100.0 / len(result)
print(f'整体准确率为：{overall_accuracy:.2f}%')

# 输出每种类别的准确率
for label, stats in category_stats.items():
    if stats['total'] > 0:
        class_accuracy = stats['correct'] / stats['total'] * 100.0
        print(f'类别 {label} 的准确率为：{class_accuracy:.2f}%')
    else:
        print(f'类别 {label} 的准确率无法计算，因为没有样本。')