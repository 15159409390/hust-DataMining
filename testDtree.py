import csv
from Dtree import main
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

def load_data(file_path):
    with open(file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        data = [list(map(float, row)) for row in reader]
    data = np.array(data)  # 转换为numpy数组
    X = data[:, :-1]  # 特征
    y = data[:, -1]  # 标签
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=None):
    random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    test_indices = random.sample(range(n_samples), n_test)
    train_indices = [i for i in range(n_samples) if i not in test_indices]
    train_X = [X[i] for i in train_indices]
    train_y = [y[i] for i in train_indices]
    test_X = [X[i] for i in test_indices]
    test_y = [y[i] for i in test_indices]
    return train_X, train_y, test_X, test_y


def test_model():
    # 加载数据集
    X, y = load_data('span_pub2.csv')
    # 划分训练集和测试集
    train_X, train_y, test_X, test_y = train_test_split(X, y, test_size=0.89, random_state=32)
    # 训练模型并进行预测
    predictions = main(train_X, train_y, test_X)
    return  

def evaluate_performance(predictions, test_y):
    # 计算混淆矩阵
    cm = confusion_matrix(test_y, predictions)
    print("Confusion Matrix:")
    print(cm)

    # 计算精确度
    precision = precision_score(test_y, predictions)
    print("Precision:", precision)

    # 计算召回率
    recall = recall_score(test_y, predictions)
    print("Recall:", recall)

    # 计算F1分数
    f1 = f1_score(test_y, predictions)
    print("F1 Score:", f1)


if __name__ == "__main__":
    predictions, test_y = test_model()
    # print("Predictions:", predictions)
    # print("Actual:", test_y)
    
    # 计算准确性
    accuracy = sum(1 for pred, actual in zip(predictions, test_y) if pred == actual) / len(test_y)
    print("Accuracy:", accuracy)

    # 计算其他性能指标
    evaluate_performance(predictions, test_y)


