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
【例6-4】通过相机标定后，进行坐标系转换，得图像中的三维坐标。最后计算图像中两个三维点之间的距离，并显示结果。
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

# 对第一张图像进行坐标系转换
img = cv2.imread(images[0])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 取第一个角点和最后一个角点
u1, v1 = imgpoints[0][0].ravel()
u2, v2 = imgpoints[0][-1].ravel()

# 定义函数将二维像素坐标转换为三维世界坐标
def to_3D(u, v, K, R, t):
    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)
    R_inv_T = np.dot(R_inv, t)
    coords = np.array([[u], [v], [1]])

    cam_point = np.dot(K_inv, coords)
    cam_R_inv = np.dot(R_inv, cam_point)
    scale = R_inv_T[2][0] / cam_R_inv[2][0]
    scale_world = scale * cam_R_inv
    world_point = scale_world - R_inv_T
    pt = np.zeros((3, 1), dtype=np.float64)
    pt[0] = world_point[0]
    pt[1] = world_point[1]
    pt[2] = 0
    pt_arr = np.asarray(pt).squeeze()

    return pt_arr

# 计算三维世界坐标
rvec, tvec = rvecs[0], tvecs[0]
R, _ = cv2.Rodrigues(rvec)
world_coords1 = to_3D(u1, v1, mtx, R, tvec)
world_coords2 = to_3D(u2, v2, mtx, R, tvec)

# 计算两点之间的三维距离
distance = np.linalg.norm(world_coords1 - world_coords2)

# 绘制角点
scaled_u1, scaled_v1 = int(u1), int(v1)
scaled_u2, scaled_v2 = int(u2), int(v2)
cv2.circle(img, (int(u1), int(v1)), 5, (0, 255, 0), -1)
cv2.circle(img, (int(u2), int(v2)), 5, (0, 0, 255), -1)
cv2.line(img, (int(u1), int(v1)), (int(u2), int(v2)), (255, 0, 0), 2)

# 绘制文本
cv2.putText(img, f'Distance between the two 3D points: {distance:.5f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(img, f'({u1:.5f}, {v1:.5f})', (scaled_u1, scaled_v1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 60, 60), 2, cv2.LINE_AA)
cv2.putText(img, f'({u2:.5f}, {v2:.5f})', (scaled_u2, scaled_v2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 60, 60), 2, cv2.LINE_AA)
cv2.putText(img, f'({world_coords1[0]:.5f}, {world_coords1[1]:.5f})', (scaled_u1, scaled_v1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2, cv2.LINE_AA)
cv2.putText(img, f'({world_coords2[0]:.5f}, {world_coords2[1]:.5f})', (scaled_u2, scaled_v2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2, cv2.LINE_AA)

# 显示图像
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()