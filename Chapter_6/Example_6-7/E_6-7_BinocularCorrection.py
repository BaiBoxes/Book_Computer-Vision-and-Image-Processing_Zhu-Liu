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
【例6-7】使用cv2.stereoRectify函数计算校正变换矩阵，之后利用cv2.initUndistortRectifyMap函数生成校正映射，再cv2.remap函数进行校正映射，最后保留并对比校正前后的图像。
"""


import numpy as np
import cv2
import glob


# 标定部分
# 读取左相机和右相机的图片
images_left = glob.glob('image/Chessboard_dual/left/*.png')
images_right = glob.glob('image/Chessboard_dual/right/*.png')

# 设置棋盘格参数，内角点个数与每个方格的实际大小
chessboard_size = (8, 6)
square_size = 0.03

# 准备棋盘格的世界坐标系坐标
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储所有图像的对象点和图像点
objpoints = []
imgpoints_left = []
imgpoints_right = []

if not images_left or not images_right:
    print("没有找到指定目录中的图片。")
else:
    print(f"找到 {len(images_left)} 张左图像和 {len(images_right)} 张右图像。")

for img_left, img_right in zip(images_left, images_right):
    # 读取图片
    imgL = cv2.imread(img_left)
    imgR = cv2.imread(img_right)

    if imgL is None or imgR is None:
        print(f"无法读取图片 {img_left} 或 {img_right}")
        continue

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    retL, cornersL = cv2.findChessboardCorners(grayL, chessboard_size, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, chessboard_size, None)

    if retL and retR:
        objpoints.append(objp)
        cornersL2 = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints_left.append(cornersL2)

        cornersR2 = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints_right.append(cornersR2)

        # 可视化角点检测结果
        cv2.drawChessboardCorners(imgL, chessboard_size, cornersL2, retL)
        cv2.drawChessboardCorners(imgR, chessboard_size, cornersR2, retR)

        # 显示拼接后的图像
        concatenated_img = cv2.hconcat([imgL, imgR])
        cv2.imshow('Concatenated Image', concatenated_img)
        cv2.waitKey(100)
    else:
        print(f"未能在图片 {img_left} 或 {img_right} 中找到角点")

# 关闭所有显示窗口
cv2.destroyAllWindows()

# 单目标定
if len(objpoints) > 0 and len(imgpoints_left) > 0:
    retL, cameraMatrixL, distCoeffsL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpoints_left, grayL.shape[::-1], None, None)
else:
    print("左相机校准的有效图片数量不足")

if len(objpoints) > 0 and len(imgpoints_right) > 0:
    retR, cameraMatrixR, distCoeffsR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpoints_right, grayR.shape[::-1], None, None)
else:
    print("右相机校准的有效图片数量不足")

# 双目标定
if len(objpoints) > 0 and len(imgpoints_left) > 0 and len(imgpoints_right) > 0:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, cameraMatrixL, distCoeffsL, cameraMatrixR, distCoeffsR, grayL.shape[::-1], R=None, T=None, E=None, F=None, flags=cv2.CALIB_FIX_INTRINSIC, criteria=criteria)
    
else:
    print("双目校准的有效图片数量不足")

# 双目校正
if len(objpoints) > 0 and len(imgpoints_left) > 0 and len(imgpoints_right) > 0:
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, grayL.shape[::-1], R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, grayL.shape[::-1], cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, grayR.shape[::-1], cv2.CV_32FC1)

    # 显示原图像与校正后的图像
    for idx, (img_left, img_right) in enumerate(zip(images_left, images_right)):

        # 读取图像
        imgL = cv2.imread(img_left)
        imgR = cv2.imread(img_right)

        # 校正图像
        rectifiedL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectifiedR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

        # 水平拼接原图像和校正后的图像
        concat_top = cv2.hconcat([imgL, imgR])
        concat_bottom = cv2.hconcat([rectifiedL, rectifiedR])

        # 垂直拼接水平拼接后的图像
        concatenated_img = cv2.vconcat([concat_top, concat_bottom])

        # 显示拼接后的图像
        cv2.imshow('Concatenated Image', concatenated_img)
        cv2.waitKey(1000)

        cv2.destroyAllWindows()
else:
    print("双目校正的有效图片数量不足")