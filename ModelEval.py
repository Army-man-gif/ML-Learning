from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()
tree = RandomForestRegressor(random_state=0)
X2, y2 = load_iris(return_X_y=True)
clf = RandomForestClassifier(random_state=0)

result = cross_validate(lr, X, y)  # defaults to 5-fold CV if one not specified
result2 = cross_validate(tree, X, y, cv=5)
result3 = cross_validate(clf, X2, y2, cv=5)

score = result['test_score']  # r_squared score is high because dataset is easy
score2 = result2['test_score']
score3 = result3['test_score'] 
 
print(score)
print(score2)
print(score3)