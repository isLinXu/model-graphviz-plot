import matplotlib.pyplot as plt

# 创建 Figure 对象
fig = plt.figure(figsize=(10, 8))

# 创建 Axes 对象
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

# 绘制网络层
ax.text(0.5, 0.9, 'Input', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.8, 'Conv2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.7, 'MaxPooling2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.6, 'Conv2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.5, 'MaxPooling2D', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.4, 'Flatten', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.3, 'Dense', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.2, 'Dense', ha='center', va='center', fontsize=14, fontweight='bold')
ax.text(0.5, 0.1, 'Output', ha='center', va='center', fontsize=14, fontweight='bold')

# 绘制连接线
ax.plot([0.5, 0.5], [0.88, 0.82], 'k-', lw=2)
ax.plot([0.5, 0.5], [0.78, 0.72], 'k-', lw=2)
ax.plot([0.5, 0.5], [0.68, 0.62], 'k-', lw=2)
ax.plot([0.5, 0.5], [0.58, 0.52], 'k-', lw=2)
ax.plot([0.5, 0.5], [0.48, 0.42], 'k-', lw=2)
ax.plot([0.5, 0.5], [0.38, 0.32], 'k-', lw=2)
ax.plot([0.5, 0.5], [0.28, 0.22], 'k-', lw=2)
ax.plot([0.5, 0.5], [0.18, 0.12], 'k-', lw=2)

# 设置属性
ax.set_title('Deep Learning Model', fontsize=16, fontweight='bold')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xticks([])
ax.set_yticks([])

# 显示图像
plt.show()