import numpy as np
import pandas as pd
import sklearn.model_selection as sl
import sklearn.neighbors as sn
import sklearn.preprocessing as sk

#Загрузка выборки Wine из https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
data = pd.read_csv('wine.data', header=None)

#Извлекаются из данных признаки и классы. Класс записан в первом столбце (три варианта), признаки — в столбцах со второго по последний. Более подробно о сути признаков можно прочитать по адресу https://archive.ics.uci.edu/ml/datasets/Wine 
X = pd.read_csv('wine.data', header=None, usecols=list(range(1,14)))
y = pd.read_csv('wine.data', header=None, usecols=[0]).values.reshape(len(X),)

#Оценку качества необходимо провести методом кросс-валидации по 5 блокам (5-fold). Создается генератор разбиений, который перемешивает выборку перед формированием блоков (shuffle=True). Для воспроизводимости результата, создается генератор KFold с фиксированным параметром random_state=42. В качестве меры качества используйте долю верных ответов (accuracy).
kf = sl.KFold(n_splits=5, shuffle=True, random_state=42)

#Находится точность классификации на кросс-валидации для метода k ближайших соседей (sklearn.neighbors.KNeighborsClassifier), при k от 1 до 50
kMeans = list()
for k in range(1, 51):
    kn = sn.KNeighborsClassifier(n_neighbors=k)
    kn.fit(X, y);
    array = sl.cross_val_score(estimator=kn, X=X, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m) 

m = max(kMeans)
indices = [i for i, j in enumerate(kMeans) if j == m]

#Вывод, при каком k получилось оптимальное качество (число в интервале от 0 до 1) 
print (indices[0]+1)
print (np.round(m,decimals=2))
 
#Производится масштабирование признаков с помощью функции sklearn.preprocessing.scale. 
X_scale = sk.scale(X)
 
#Снова находится оптимальное k на кросс-валидации. 
kMeans = list()
for k in range(1, 51):
    kn = sn.KNeighborsClassifier(n_neighbors=k)
    array = sl.cross_val_score(estimator=kn, X=X_scale, y=y, cv=kf, scoring='accuracy')
    m = array.mean()
    kMeans.append(m)   
 
m = max(kMeans)
indices = [i for i, j in enumerate(kMeans) if j == m]

#Вывод, какое значение k получилось оптимальным после приведения признаков к одному масштабу 
print (indices[0]+1)
print (np.round(m,decimals=2))