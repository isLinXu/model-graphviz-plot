import matplotlib.pyplot as plt
from keras.applications import ResNet50, VGG16
from keras.utils import plot_model

# 加载预训练的ResNet50模型
resnet50_model = ResNet50(weights='imagenet', include_top=True)

# 使用plot_model函数绘制ResNet50模型结构，并保存为图片
plot_model(resnet50_model, to_file='resnet50_model.png', show_shapes=True, show_layer_names=True)

# 加载预训练的VGG16模型
vgg16_model = VGG16(weights='imagenet', include_top=True)

# 使用plot_model函数绘制VGG16模型结构，并保存为图片
plot_model(vgg16_model, to_file='vgg16_model.png', show_shapes=True, show_layer_names=True)

# 显示ResNet50模型结构图片
resnet50_img = plt.imread('resnet50_model.png')
plt.figure(figsize=(20, 40))
plt.axis('off')
plt.imshow(resnet50_img)
plt.show()

# 显示VGG16模型结构图片
vgg16_img = plt.imread('vgg16_model.png')
plt.figure(figsize=(20, 40))
plt.axis('off')
plt.imshow(vgg16_img)
plt.show()