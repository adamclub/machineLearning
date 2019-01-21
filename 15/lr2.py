#lr.py
#21 天入门机器学习-第02期
#第15课：逻辑回归——用来做分类的回归模型
#示例代码
'''
例子，比如某位老师想用学生上学期考试的成绩（Last Score）和本学期在学习上花费的时间（Hours Spent）来预期本学期的成绩

我们把前11个样本作为训练集，最后3个样本作为测试集。

这样训练出来之后，得到的预测结果为：[55.33375602 54.29040467 90.76185124]，也就说 id 为 12-14 的三个同学的预测分数为55，54和91。

第一个差别比较大，id 为12的同学，明明考及格了，却被预测为不及格。

这是为什么呢？大家注意 id 为4的同学，这是一位学霸，他只用了20小时在学习上，却考出了第一名的好成绩。

回想一下线性回归的目标函数，我们不难发现，所有训练样本对于目标的贡献是平均的，因此，4号同学这种超常学霸的出现，在数据量本身就小的情况下，有可能影响整个模型。

这还是幸亏我们有历史记录，知道上次考试的成绩，如果 X 只包含“Hours Spent”，学霸同学根本就会带偏大多数的预测结果（自变量只有“Hours Spent”的线性回归模型会是什么样的？这个问题留给同学们自己去实践）。

那么我们看看用逻辑回归如何。用逻辑回归的时候，我们就不再是预测具体分数，而是预测这个学生本次能否及格了。

这样我们就需要对数据先做一下转换，把具体分数转变成是否合格，合格标志为1，不合格为0，然后再进行逻辑回归：
'''
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd

# Importing dataset
data = pd.read_csv('quiz.csv', delimiter=',')

used_features = [ "Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values

X_train = X[:11]
X_test = X[11:]

# Logistic Regression – Binary Classification
passed = []

for i in range(len(scores)):
    if(scores[i] >= 60):
        passed.append(1)
    else:
        passed.append(0)

y_train = passed[:11]
y_test = passed[11:]

#why C=1e5? α?
classifier = LogisticRegression(C=1e5)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
print(y_predict)
