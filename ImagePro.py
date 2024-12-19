import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 定义文件夹路径
folder1 = "D:/lund/Machine Learning/machine learning project/machine learning project/Raw_Photos(train_val)/trainyes"  # 标签为1的文件夹
folder2 = "D:/lund/Machine Learning/machine learning project/machine learning project/Raw_Photos(train_val)/trainno"  # 标签为0的文件夹

# 加载图片并添加标签
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            img = Image.open(file_path).convert('RGB')  # 转为RGB格式
            img = img.resize((128, 128))  # 调整到统一大小
            images.append(np.array(img))
            labels.append(label)  # 分配标签
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    return np.array(images), np.array(labels)

# 加载两个文件夹的图片和标签
images1, labels1 = load_images_from_folder(folder1, label=1)
images2, labels2 = load_images_from_folder(folder2, label=0)

# 合并数据集和标签
images = np.concatenate((images1, images2), axis=0)
labels = np.concatenate((labels1, labels2), axis=0)

# 检查图片和标签形状
print(f"Images shape: {images.shape}")  # e.g., (num_images, 128, 128, 3)
print(f"Labels shape: {labels.shape}")  # e.g., (num_images,)

# 归一化处理 (将像素值缩放到 [0, 1])
images_normalized = images / 255.0
print("Normalization complete!")

# 将图片展平 (每张图片变为一维向量)
num_images, height, width, channels = images_normalized.shape
images_flattened = images_normalized.reshape(num_images, -1)
print(f"Flattened shape: {images_flattened.shape}")  # e.g., (num_images, 128*128*3)

# 标准化数据 (零均值单位方差)
scaler = StandardScaler()
images_standardized = scaler.fit_transform(images_flattened)
print("Standardization complete!")

# 线性降维 (PCA)
pca = PCA(n_components=75)  # 降维到75维
images_pca = pca.fit_transform(images_standardized)
print(f"PCA completed! Reduced shape: {images_pca.shape}")

# 合并降维后的数据和标签
processed_data = np.hstack((images_pca, labels.reshape(-1, 1)))

# 保存降维后的数据
np.save("processed_images_with_labels.npy", processed_data)
print("Processed images with labels saved!")
