import matplotlib.pyplot as plt

# 创建 Figure 对象
fig = plt.figure(figsize=(12, 8))

# 创建 Axes 对象
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# 绘制网络层
ax.text(0.1, 0.95, 'Input', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.3, 0.95, 'Conv2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.95, 'MaxPooling2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.7, 0.95, 'Conv2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.9, 0.95, 'MaxPooling2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.85, 'Flatten', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.3, 0.75, 'Dense', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.7, 0.75, 'Dense', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.65, 'Output', ha='center', va='center', fontsize=14, fontweight='bold')

# 绘制连接线
ax.plot([0.1, 0.3], [0.95, 0.95], 'r--', lw=2)
ax.plot([0.3, 0.5], [0.95, 0.95], 'b-', lw=2)
ax.plot([0.5, 0.7], [0.95, 0.95], 'b-', lw=2)
ax.plot([0.7, 0.9], [0.95, 0.95], 'b-', lw=2)
ax.plot([0.5, 0.5], [0.85, 0.75], 'g-', lw=2)
ax.plot([0.3, 0.5], [0.75, 0.75], 'b-', lw=2)
ax.plot([0.7, 0.9], [0.75, 0.75], 'b-', lw=2)
ax.plot([0.5, 0.5], [0.75, 0.65], 'g-', lw=2)

# 添加网络层的参数和输出大小等信息
ax.text(0.3, 0.9, '32 filters\n(3, 3) kernel\nReLU activation', ha='center', va='center', fontsize=12)
ax.text(0.5, 0.9, '2x2 pool size', ha='center', va='center', fontsize=12)
ax.text(0.7, 0.9, '64 filters\n(3, 3) kernel\nReLU activation', ha='center', va='center', fontsize=12)
ax.text(0.3, 0.7, '256 units\nReLU activation', ha='center', va='center', fontsize=12)
ax.text(0.7, 0.7, '10 units\nSoftmax activation', ha='center', va='center', fontsize=12)
ax.text(0.5, 0.6, '10 classes', ha='center', va='center', fontsize=12)

# 设置属性
ax.set_title('Convolutional Neural Network', fontsize=16, fontweight='bold')
ax.set_xlim([0, 1])
ax.set_ylim([0,1])
ax.set_xticks([])
ax.set_yticks([])

plt.show()