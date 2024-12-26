import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # 划分特征的索引
        self.threshold = threshold  # 划分阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶节点的类别


class Dtree:
    def __init__(self, max_depth=None, min_samples_leaf=None):
        self.max_depth = max_depth  # 最大深度限制
        self.min_samples_leaf = min_samples_leaf  # 叶节点最小样本数
        self.tree = None

    def fit(self, X, Y):
        X = np.array(X)  # 转换为numpy数组
        Y = np.array(Y)  # 转换为numpy数组
        self.tree = self._grow_tree(X, Y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # 如果达到停止条件之一，则返回叶节点
        if n_labels == 1 or n_samples < self.min_samples_leaf or depth == self.max_depth:
            return Node(value=max(Counter(y).items(), key=lambda x: x[1])[0])

        # 选择最优划分特征和阈值
        best_feature, best_threshold = self._best_split(X, y, n_samples, n_features)

        # 如果无法再划分，则返回叶节点
        if best_feature is None:
            return Node(value=max(Counter(y).items(), key=lambda x: x[1])[0])

        # 递归构建左右子树
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        left = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold, left=left, right=right)


    def _best_split(self, X, y, n_samples, n_features):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            thresholds = sorted(set(X[:, feature_index]))

            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = ~left_indices

                gini = self._gini_impurity(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, left_labels, right_labels):
        total_samples = len(left_labels) + len(right_labels)
        p_left = len(left_labels) / total_samples
        p_right = len(right_labels) / total_samples

        unique_left, counts_left = np.unique(left_labels, return_counts=True)
        unique_right, counts_right = np.unique(right_labels, return_counts=True)

        gini_left = 1 - np.sum((counts_left / len(left_labels)) ** 2)
        gini_right = 1 - np.sum((counts_right / len(right_labels)) ** 2)

        gini = p_left * gini_left + p_right * gini_right

        return gini


    def predict(self, X):
        X = np.array(X)  # 转换为numpy数组
        return [self._predict_one(sample, self.tree) for sample in X]


    def _predict_one(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] < node.threshold:
            return self._predict_one(sample, node.left)
        else:
            return self._predict_one(sample, node.right)


def main(train_X, train_Y, test_X):
    dtree = Dtree(max_depth=5, min_samples_leaf=5)
    dtree.fit(train_X, train_Y)
    return dtree.predict(test_X)



