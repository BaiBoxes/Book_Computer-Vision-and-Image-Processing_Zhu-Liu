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
【例5-11】一个一维的运动轨迹，0到5范围。在第n帧时，识别到目标在3位置上，可信度为0.96，然后2与4位置上，可信度分别为0.91与0.85，使用二次插值对第n帧的运动估计位置进行亚像素化。
"""


import numpy as np
import matplotlib.pyplot as plt


# 定义已知的位置和可信度
positions = np.array([2, 3, 4])
confidence = np.array([0.91, 0.96, 0.85])

# 拟合二次多项式
coefficients = np.polyfit(positions, confidence, 2)

# 定义多项式
polynomial = np.poly1d(coefficients)

# 计算二次多项式的导数
derivative = polynomial.deriv()

# 求解导数为零的位置，即最大值的位置
max_position = np.roots(derivative)[0]
print(f"第 n 帧时，亚像素化后目标的位置为: {max_position:.4f}")

# 计算该位置的最大可信度
max_confidence = polynomial(max_position)

# 绘制多项式曲线和数据点
x = np.linspace(0, 5, 500)
y = polynomial(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Quadratic Fit')
plt.scatter(positions, confidence, color='red', label='Data Points')
plt.axhline(max_confidence, color='green', linestyle='--', label=f'Max Confidence = {max_confidence:.4f}')
plt.scatter(max_position, max_confidence, color='blue', label=f'Max Position={max_position:.4f}')
plt.xlabel('Position')
plt.ylabel('Confidence')
plt.title('Quadratic Polynomial Fit with Maximum Confidence')
plt.legend()
plt.grid(True)
plt.show()