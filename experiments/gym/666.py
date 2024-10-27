import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机数据
x = np.random.randn(1000)
y = np.random.randn(1000)

# 计算二维直方图
H, xedges, yedges = np.histogram2d(x, y, bins=30)

# 调整直方图显示范围的参数（extent）
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# 绘制二维直方图
plt.imshow(H.T, origin="lower", cmap="viridis", aspect='auto', extent=extent)
plt.colorbar(label='Counts')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Histogram')
plt.show()
