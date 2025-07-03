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
【例10-2】使用k近邻算法对train_image文件夹中的形状图片进行训练，再对predict_image文件夹中的图片进行预测，最后直观的展示结果和模型的准确率。
"""


import numpy as np
import cv2
import os


def load_images(folder):
    test_images = []
    test_labels = []
    filenames = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # 保持图像原始100x100的尺寸
            img_resized = img.reshape((1, 10000))
            test_images.append(img_resized)
            filenames.append(filename)

            # 假设测试标签与训练标签的生成规则相同
            label = filename.split('_')[0]
            if label == 'circle':
                test_labels.append(0)
            elif label == 'square':
                test_labels.append(1)
            elif label == 'star':
                test_labels.append(2)
            elif label == 'triangle':
                test_labels.append(3)

    return np.array(test_images, dtype='float32'), np.array(test_labels, dtype='int32'), filenames



# 模式选择
mode = input("请输入模式 ('train' 或 'predict'): ").strip().lower()

if mode == 'train':
    # 训练模型部分
    train_images, train_labels, _ = load_images('image\\Learn\\train_image')

    # 将数据整合为KNN输入格式
    train_data = train_images.reshape(-1, 10000).astype(np.float32)

    # 创建KNN模型
    knn = cv2.ml.KNearest_create()
    knn.setDefaultK(5)

    # 训练模型
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    # 保存训练模型和数据
    cv2.imwrite('./Chapter_9/Example_9-2/results/train_data.png', train_data)
    cv2.imwrite('./Chapter_9/Example_9-2/results/train_labels.png', train_labels)
    knn.save('./Chapter_9/Example_9-2/results/knn_model.yml')
    print('模型训练完成并已保存。')

elif mode == 'predict':
    # 加载KNN模型
    knn = cv2.ml.KNearest_load('./Chapter_9/Example_9-2/results/knn_model.yml')

    # 加载预测数据
    test_images, test_labels, filenames = load_images('image\\Learn\\predict_image')
    test_data = test_images.reshape(-1, 10000).astype(np.float32)

    # 进行预测
    ret, results, neighbours, dist = knn.findNearest(test_data, k=1)

    # 记录每种类别的正确预测和总预测数
    category_stats = {
        0: {'correct': 0, 'total': 0},  # circle
        1: {'correct': 0, 'total': 0},  # square
        2: {'correct': 0, 'total': 0},  # star
        3: {'correct': 0, 'total': 0}   # triangle
    }

    correct_count = 0
    for i in range(len(results)):
        predicted_label = int(results[i][0])
        actual_label = int(test_labels[i][0])

        # 更新每种类别的统计信息
        category_stats[actual_label]['total'] += 1
        if predicted_label == actual_label:
            category_stats[actual_label]['correct'] += 1
            correct_count += 1

        print('图像 {} 的预测类别为：{}，实际类别为：{}'.format(filenames[i], predicted_label, actual_label))

    accuracy = correct_count / len(results)
    print(f'预测完成。整体准确率为：{accuracy:.2%}')

    # 输出每种类别的准确率
    for label, stats in category_stats.items():
        if stats['total'] > 0:
            class_accuracy = stats['correct'] / stats['total']
            print(f'类别 {label} 的准确率为：{class_accuracy:.2%}')
        else:
            print(f'类别 {label} 的准确率无法计算，因为没有样本。')

else:
    print("无效的模式输入，请输入 'train' 或 'predict'。")