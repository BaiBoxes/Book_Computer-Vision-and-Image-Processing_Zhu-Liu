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
【例6-3】假定一组三维点坐标，使用OpenCV库中的cv2.projectPoints函数将三维世界中的点投影到二维图像平面上，展示投影效果，并保存。
"""


import numpy as np
import cv2
import glob


# 单目标定
# 读取目录下的所有图像
images = glob.glob('image//Chessboard_single//*.png')

# 设置棋盘格参数，内角点个数与每个方格的实际大小
chessboard_size = (8, 6)
square_size = 0.03

# 准备棋盘格的世界坐标系坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储所有图像的对象点和图像点
objpoints = []
imgpoints = []

for fname in images:
    if len(images) == 0:
        raise ValueError("未找到任何图像，请检查图像路径或图像文件是否存在")
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    # 如果找到了，添加对象点，图像点
    if ret:
        objpoints.append(objp)
        # 精确化角点位置
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # 绘制并显示角点
        img = cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        # 显示图像
        cv2.imshow('img', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()

# 检查objpoints和imgpoints是否为空
if len(objpoints) == 0 or len(imgpoints) == 0:
    raise ValueError("未找到足够的棋盘格角点，无法进行相机标定")

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 读取单张棋盘格图像进行投影
img = cv2.imread(images[0])

# 随机生成6个三维空间中的点
object_points = np.random.uniform(-0.1, 0.2, size=(6, 3))  

# 使用第一个图像的旋转向量和平移向量进行投影
rvec, tvec = rvecs[0], tvecs[0]

# 使用cv2.projectPoints函数进行单目投影
image_points, _ = cv2.projectPoints(object_points, rvec, tvec, mtx, dist)

# 绘制投影点
for point in image_points:
    x, y = int(point[0][0]), int(point[0][1])
    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    cv2.putText(img, f'({point[0][0]:.5f}, {point[0][1]:.5f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2, cv2.LINE_AA)

# 评估重投影误差
total_error = 0
for i in range(len(objpoints)):
    # 将世界坐标下的点投影到图像坐标系
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) ** 2 / len(imgpoints2)
    total_error += error
mean_error = np.sqrt(total_error / len(objpoints))
print(f"重投影误差: {mean_error:.5f}")

# 显示结果
cv2.imshow('Projected Points', img)
cv2.waitKey(0)
cv2.destroyAllWindows()