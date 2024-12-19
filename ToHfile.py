import numpy as np
from sklearn.model_selection import train_test_split

# 加载降维后的数据
data = np.load("processed_images_with_labels.npy")

# 分离特征和标签
X = data[:, :-1]  # 特征
y = data[:, -1]   # 标签

# 划分训练集(70%)、验证集(20%)和测试集(10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# 将数据格式化为 C 数组
def convert_to_c_array(data, name):
    c_array = f"float {name}[] = {{\n"
    for row in data:
        c_array += "    " + ", ".join(map(str, row)) + ",\n"
    c_array = c_array.rstrip(",\n") + "\n};\n"
    return c_array

def convert_labels_to_c_array(labels, name):
    c_array = f"int {name}[] = {{\n"
    c_array += "    " + ", ".join(map(str, labels.astype(int))) + "\n};\n"
    return c_array

# 转换特征和标签为 C 数组
train_features_c = convert_to_c_array(X_train, "train_features")
train_labels_c = convert_labels_to_c_array(y_train, "train_labels")
val_features_c = convert_to_c_array(X_val, "val_features")
val_labels_c = convert_labels_to_c_array(y_val, "val_labels")
test_features_c = convert_to_c_array(X_test, "test_features")
test_labels_c = convert_labels_to_c_array(y_test, "test_labels")

# 生成 .h 文件
header_content = (
    "// Auto-generated dataset header file\n\n"
    "#ifndef DATASET_H\n#define DATASET_H\n\n"
    f"// Train set: Features = {X_train.shape}, Labels = {y_train.shape}\n"
    f"{train_features_c}\n"
    f"{train_labels_c}\n"
    f"// Validation set: Features = {X_val.shape}, Labels = {y_val.shape}\n"
    f"{val_features_c}\n"
    f"{val_labels_c}\n"
    f"// Test set: Features = {X_test.shape}, Labels = {y_test.shape}\n"
    f"{test_features_c}\n"
    f"{test_labels_c}\n"
    "#endif // DATASET_H\n"
)


# 保存到 .h 文件
with open("dataset.h", "w") as f:
    f.write(header_content)

print("Header file 'dataset.h' has been generated!")
