import os
import random
import cv2
import numpy as np

# 输出目录
output_dir = "image\Learn\GAN_train"

# 定义图像尺寸
img_size = (28, 28)

# 生成细微变化的大小
def random_size(base_size, variance):
    return base_size + random.randint(-variance, variance)

# 生成图像
def create_image():
    # 创建白色背景
    img = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255

    # 绘制圆形
    radius = random_size(8, 1)  # 圆的半径在7到9之间变化
    center = (img_size[1] // 2, img_size[0] // 2)  # 圆心位于中心
    cv2.circle(img, center, radius, (0, 255, 255), -1)  # 黄色圆形

    # 绘制左侧三角形（蓝色）
    left_points = np.array([
        [0, random.randint(5, img_size[0] - 5)],  # 左顶点
        [0, random.randint(5, img_size[0] - 5)],  # 右顶点
        [random_size(8, 2), random.randint(5, img_size[0] - 5)]  # 顶点
    ])

    left_points = left_points.reshape((-1, 1, 2))  # 形状调整为 (n, 1, 2)
    cv2.fillPoly(img, [left_points], color=(255, 0, 0))  # 蓝色三角形

    # 绘制右侧三角形（绿色）
    right_points = np.array([
        [img_size[1] - 1, random.randint(5, img_size[0] - 5)],  # 左顶点
        [img_size[1] - 1, random.randint(5, img_size[0] - 5)],  # 右顶点
        [img_size[1] - random_size(8, 2), random.randint(5, img_size[0] - 5)]  # 顶点
    ])

    right_points = right_points.reshape((-1, 1, 2))  # 形状调整为 (n, 1, 2)
    cv2.fillPoly(img, [right_points], color=(0, 255, 0))  # 绿色三角形

    return img

# 创建并保存图像
def save_images(count):
    for i in range(count):
        img = create_image()
        cv2.imwrite(os.path.join(output_dir, f"image_{i + 1}.png"), img)

# 生成图像
save_images(50)

print("图像生成完成！")