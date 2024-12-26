import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV  
import matplotlib.pyplot as plt
import numpy as np
from unidecode import unidecode

def load_data(file_path: str):
    """
    加载数据集。

    参数:
    file_path (str): 数据集文件路径。

    返回:
    df (pd.DataFrame): 加载后的数据集。
    """
    # 加载数据集
    df = pd.read_csv(file_path)

    # 删除第一列无用的数据
    df.drop(columns=['Unnamed: 0'], inplace=True)

    return df

def preprocessing_data(df):
    """
    预处理数据。

    参数:
    df (pd.DataFrame): 数据集。

    返回:
    X_train (pd.DataFrame): 训练集特征。
    X_test (pd.DataFrame): 测试集特征。
    y_train (pd.Series): 训练集标签。
    y_test (pd.Series): 测试集标签。
    """
    # 划分特征和标签
    X = df.drop(columns=['spam'])
    y = df['spam']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def svm_train(X_train, y_train):
    """
    训练SVM模型。

    参数:
    X_train (pd.DataFrame): 训练集特征。
    y_train (pd.Series): 训练集标签。

    返回:
    svm_model (SVC): 训练好的SVM模型。
    """
    # 创建SVM模型
    svm_model = SVC(kernel='rbf', random_state=40)

    # 使用训练数据训练模型
    svm_model.fit(X_train, y_train)

    return svm_model

def evaluate_model(svm_model, X_test, y_test):
    """
    评估模型性能。

    参数:
    svm_model (SVC): 训练好的SVM模型。
    X_test (pd.DataFrame): 测试集特征。
    y_test (pd.Series): 测试集标签。

    返回:
    accuracy (float): 准确率。
    report (str): 分类报告。
    """
    # 在测试集上进行预测
    y_pred = svm_model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 输出分类报告
    report = classification_report(y_test, y_pred)

    return accuracy, report


def plot_roc_curve(svm_model, X_test, y_test):
    """
    绘制ROC曲线。

    参数:
    svm_model (SVC): 训练好的SVM模型。
    X_test (pd.DataFrame): 测试集特征。
    y_test (pd.Series): 测试集标签。
    """
    # 计算预测概率
    y_score = svm_model.decision_function(X_test)

    # 计算ROC曲线相关数据
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线图
    plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.xticks(rotation=45)
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC Curve')
    plt.legend()
    plt.show()

def plot_learning_curve(svm_model, X, y, kernel):
    """
    绘制学习曲线。

    参数:
    svm_model (SVC): 训练好的SVM模型。
    X (pd.DataFrame): 全量数据集特征。
    y (pd.Series): 全量数据集标签。
    """
    # 计算学习率曲线
    train_sizes, train_scores, test_scores = learning_curve(svm_model, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))

    # 计算平均训练准确率和平均测试准确率
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # 绘制学习率曲线
    plt.plot(train_sizes, train_mean, label='Training Accuracy')
    plt.plot(train_sizes, test_mean, label='Validation Accuracy')
    plt.title(f'Learning Curve (Kernel: {kernel})')
    plt.xlabel('Training Set Size')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def main():
    # 定义数据所在地址
    file_path = 'span_pub2.csv'

    # 加载数据
    df = load_data(file_path)

    # 选择SVM核函数
    kernel = 'rbf'

    # 预处理数据
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessing_data(df)

    # 训练模型
    svm_model = svm_train(X_train_scaled, y_train)

    # 输出模型准确率和分类报告
    accuracy, report = evaluate_model(svm_model, X_test_scaled, y_test)
    print(f'准确率：{accuracy}')
    print('分类结果：\n', report)

    # 绘制ROC曲线
    plot_roc_curve(svm_model, X_test_scaled, y_test)

    # 绘制学习曲线
    X = df.drop(columns=['spam'])
    y = df['spam']
    plot_learning_curve(svm_model, X, y, kernel)

if __name__ == '__main__':
    main()