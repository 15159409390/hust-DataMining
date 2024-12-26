import csv
import random
import math
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器类
    """

    def load_dataset(self, filename):
        """
        从CSV文件中加载数据集，并分离特征和标签。

        参数：
            filename (str)：数据集的文件名。

        返回：
            list：特征列表。
            list：标签列表。
        """
        with open(filename, 'r') as csv_file:
            reader = csv.reader(csv_file)
            # 获取数据集的属性名称，为数据的第一行，第二列到最后一列
            features = list(reader)[0][1:]
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            # 获取真实数据集，为数据的第二行到最后一行，第二列到最后一列
            data = list(reader)[1:]
            data = [row[1:] for row in data]

        # 将数据集转换为浮点数类型
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = float(data[i][j])

        # 将特征和标签分开
        X = [row[:-1] for row in data]
        y = [row[-1] for row in data]

        return X, y

    def train_test_split(self, X, y, test_size=0.2):
        """
        将数据集划分为训练集和测试集。

        参数：
            X (list)：特征数据集。
            y (list)：标签数据集。
            test_size (float)：测试集占总数据集的比例。

        返回：
            包含划分后的特征和标签的元组。
        """
        combined = list(zip(X, y))
        random.shuffle(combined)
        X_shuffled, y_shuffled = zip(*combined)

        # 划分数据集
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
        y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]

        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        """
        使用训练数据集训练朴素贝叶斯分类器。

        参数：
            X (list)：训练数据集特征。
            y (list)：训练数据集标签。
        """
        self.classes = list(set(y))
        self.parameters = {}

        for class_label in self.classes:
            class_data = [X[i] for i in range(len(X)) if y[i] == class_label]
            self.parameters[class_label] = {}

            for feature_index in range(len(X[0])):
                feature_values = [data[feature_index] for data in class_data]
                self.parameters[class_label][feature_index] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values)
                }

            self.parameters[class_label]['prior'] = len(class_data) / len(X)

    def calculate_gaussian_probability(self, x, mean, std):
        if std == 0:
            std = 1e-10
        exponent = math.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

    def predict(self, X):
        """
        使用训练好的模型对测试数据集进行预测。

        参数：
            X (list)：测试数据集特征。

        返回：
            list：预测的类别标签列表。
        """
        predictions = []

        for sample in X:
            posterior_probabilities = {class_label: 0 for class_label in self.classes}

            for class_label in self.classes:
                prior = self.parameters[class_label]['prior']
                posterior = prior

                for feature_index, feature_value in enumerate(sample):
                    mean = self.parameters[class_label][feature_index]['mean']
                    std = self.parameters[class_label][feature_index]['std']
                    conditional_probability = self.calculate_gaussian_probability(feature_value, mean, std)
                    # 避免概率为零的情况
                    if conditional_probability < 1e-10:
                        conditional_probability = 1e-10
                    posterior *= conditional_probability

                prior = self.parameters[class_label]['prior']
                posterior *= prior
                posterior_probabilities[class_label] = posterior

            predicted_class = max(posterior_probabilities, key=posterior_probabilities.get)
            predictions.append(predicted_class)

        return predictions

    def calculate_accuracy(self, y_true, y_pred):
        """
        计算模型预测准确率。

        参数：
            y_true (list)：真实标签列表。
            y_pred (list)：预测标签列表。

        返回：
            float：准确率，取值范围在 0 到 100 之间。
        """
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        total = len(y_true)
        accuracy = (correct / total)
        return accuracy
    
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

# 主函数
if __name__ == "__main__":
    classifier = NaiveBayesClassifier()
    X, y = classifier.load_dataset('span_pub2.csv')
    X_train, X_test, y_train, y_test = classifier.train_test_split(X, y)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    accuracy = classifier.calculate_accuracy(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 计算其他性能指标
    evaluate_performance(y_pred, y_test)