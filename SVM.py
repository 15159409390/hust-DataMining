import numpy as np
import pandas as pd
import time
import random
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # start_time = time.time()  # 记录训练开始的时间

        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]
            # 每个epoch结束后打印当前的损失
            # if (i + 1) % 100 == 0:  # 每100个epoch打印一次
            #     loss = self.calculate_loss(X, y)
            #     print(f"Epoch {i + 1}, Loss: {loss}")

        # end_time = time.time()  # 记录训练结束的时间
        # print(f"Training time: {end_time - start_time} seconds")

    def calculate_loss(self, X, y):
        loss = 0
        for idx, x_i in enumerate(X):
            condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
            if condition:
                loss += self.lambda_param * np.dot(self.w, self.w)
            else:
                loss += self.lambda_param * np.dot(self.w, self.w) + 0.5
        return loss / len(X)
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(['spam'], axis=1).values
    y = data['spam'].values
    return X, y

def shuffle_data(X, y):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

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

# Load and prepare data
X, y = load_data('span_pub2.csv')
X, y = shuffle_data(X, y)

# Split data into training and test sets
n_samples = len(X)
n_train = int(n_samples * 0.8)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# 存储最佳参数和最佳准确度
# best_acc = 0
# best_params = {}

# for i in range(10):
#     lr = random.uniform(0.001, 1)
#     lam = random.uniform(0.01, 1)
#     svm = SVM(learning_rate=lr, lambda_param=lam)
#     svm.fit(X_train, y_train)
#     y_pred = svm.predict(X_test)
#     acc = accuracy(y_test, y_pred)
#     print(f"Learning Rate: {lr}, Lambda Param: {lam}, Accuracy: {acc}")

#     if acc > best_acc:
#         best_acc = acc
#         best_params = {'learning_rate': lr, 'lambda_param': lam}

# Train SVM model
svm = SVM()
svm.fit(X_train, y_train)

# Make predictions
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy_score = accuracy(y_test, y_pred)
print(f"Accuracy: {accuracy_score}")
# 计算其他性能指标
evaluate_performance(y_pred, y_test)