#l13-Naive-Bayes-classifier-sample.py
#21 天入门机器学习-第02期
#第13课：朴素贝叶斯分类器——条件概率的参数估计
#示例代码
"""
下列数据直接存储为 career_data.csv 文件：

no,985,education,skill,enrolled

1,Yes,bachlor,C++,No

2,Yes,bachlor,Java,Yes

3,No,master,Java,Yes

4,No,master,C++,No

5,Yes,bachlor,Java,Yes

6,No,master,C++,No

7,Yes,master,Java,Yes

8,Yes,phd,C++,Yes

9,No,phd,Java,Yes

10,No,bachlor,Java,No

数据和脚本放在统一路径下，运行脚本，输出如下：

Number of mislabeled points out of a total 1 points : 0, performance 100.00%
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Importing dataset. 
# Please refer to the 【Data】 part after the code for the data file.
data = pd.read_csv("career_data.csv") 

# Convert categorical variable to numeric
data["985_cleaned"]=np.where(data["985"]=="Yes",1,0)
data["education_cleaned"]=np.where(data["education"]=="bachlor",1,
                                  np.where(data["education"]=="master",2,
                                           np.where(data["education"]=="phd",3,4)
                                          )
                                 )
data["skill_cleaned"]=np.where(data["skill"]=="c++",1,
                                  np.where(data["skill"]=="java",2,3
                                          )
                                 )
data["enrolled_cleaned"]=np.where(data["enrolled"]=="Yes",1,0)

# Split dataset in training and test datasets
X_train, X_test = train_test_split(data, test_size=0.1, random_state=int(time.time()))

# Instantiate the classifier
gnb = GaussianNB()
used_features =[
    "985_cleaned",
    "education_cleaned",
    "skill_cleaned"
]

# Train classifier
gnb.fit(
    X_train[used_features].values,
    X_train["enrolled_cleaned"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["enrolled_cleaned"] != y_pred).sum(),
          100*(1-(X_test["enrolled_cleaned"] != y_pred).sum()/X_test.shape[0])
))