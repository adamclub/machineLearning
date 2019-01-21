#lr.py
#21 天入门机器学习-第02期
#第15课：逻辑回归——用来做分类的回归模型
#示例代码
#例子，比如某位老师想用学生上学期考试的成绩（Last Score）和本学期在学习上花费的时间（Hours Spent）来预期本学期的成绩

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
