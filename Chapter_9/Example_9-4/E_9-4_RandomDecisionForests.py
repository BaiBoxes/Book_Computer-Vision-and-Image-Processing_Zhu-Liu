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
【例10-4】使用随机森林算法对2000张几何形状图像进行分类训练和预测，最后直观的展示分类的准确率和预测结果。
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
            img = cv2.resize(img, (20, 20)).flatten()
            images.append(img)
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
train_data, train_labels, _ = load_images('image\\Learn\\train_image')

# 创建随机森林模型
rt = cv2.ml.RTrees_create()
rt.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01))
rt.setMaxDepth(20)
rt.setMinSampleCount(10)
rt.setCVFolds(0)
rt.setCalculateVarImportance(True)
rt.setActiveVarCount(4)
rt.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# 保存模型到result文件夹
rt.save('./Chapter_9/Example_9-4/results/rtrees_model.xml')
print('模型训练完成并已保存。')

# 计算训练准确率
_, train_results = rt.predict(train_data)
train_matches = train_results == train_labels.reshape(-1, 1)
train_correct = np.count_nonzero(train_matches)
train_accuracy = train_correct * 100.0 / len(train_results)
print(f'RTrees模型训练的准确率为{train_accuracy:.2f}%')

# 加载模型进行预测
loaded_rt = cv2.ml.RTrees_load('./Chapter_9/Example_9-4/results/rtrees_model.xml')

# 准备测试数据
test_data, test_labels, filenames = load_images('image\\Learn\\predict_image')

# 进行预测
_, result = loaded_rt.predict(test_data)

# 初始化每个类别的统计信息
category_stats = {0: {'correct': 0, 'total': 0},
                  1: {'correct': 0, 'total': 0},
                  2: {'correct': 0, 'total': 0},
                  3: {'correct': 0, 'total': 0}}

# 更新每种类别的统计信息
for i in range(len(result)):
    predicted_label = int(result[i, 0])
    actual_label = int(test_labels[i][0])
    category_stats[actual_label]['total'] += 1
    if predicted_label == actual_label:
        category_stats[actual_label]['correct'] += 1

    print('图像 {} 的预测类别为：{}，实际类别为：{}'.format(filenames[i], predicted_label, actual_label))

# 计算预测准确率
matches = result == test_labels.reshape(-1, 1)
correct = np.count_nonzero(matches)
accuracy = correct * 100.0 / len(result)
print(f'RTrees模型预测的整体准确率为{accuracy:.2f}%')

# 输出每种类别的准确率
for label, stats in category_stats.items():
    if stats['total'] > 0:
        class_accuracy = stats['correct'] / stats['total'] * 100.0
        print(f'类别 {label} 的准确率为：{class_accuracy:.2f}%')
    else:
        print(f'类别 {label} 的准确率无法计算，因为没有样本。')