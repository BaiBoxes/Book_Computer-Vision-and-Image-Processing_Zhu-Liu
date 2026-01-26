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
【例6-2】分别使用两种校正方法对所给图像进行校正：方法一：使用cv2.initUndistortRectifyMap函数计算出校正图像所需要的映射矩阵，之后利用cv2.remap函去除原图像中的畸变；方法二：直接使用cv2.undistort函数进行校正，最后对比两种方法的校正效果，并展示校正前后的图像。
"""


import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


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

# 校正一幅图进行展示
img = cv2.imread(images[4])
h, w = img.shape[:2]

# 方法一：使用 initUndistortRectifyMap 和 remap
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, None, newCameraMatrix, (w, h), cv2.CV_16SC2)
undistorted_img1 = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

# 方法二：使用 undistort
undistorted_img2 = cv2.undistort(img, mtx, dist, None, newCameraMatrix)

# 裁剪图像以去除无效区域
x, y, w, h = roi
undistorted_img1 = undistorted_img1[y:y + h, x:x + w]
undistorted_img2 = undistorted_img2[y:y + h, x:x + w]

# 显示图像
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(wspace=0.05, hspace=0.2)
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[1].imshow(cv2.cvtColor(undistorted_img1, cv2.COLOR_BGR2RGB))
axes[1].set_title('Undistorted Image - Method 1')
axes[2].imshow(cv2.cvtColor(undistorted_img2, cv2.COLOR_BGR2RGB))
axes[2].set_title('Undistorted Image - Method 2')

for ax in axes.flat:
    ax.axis('off')

plt.show()