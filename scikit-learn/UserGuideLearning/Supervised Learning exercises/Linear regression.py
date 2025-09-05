from sklearn import linear_model
reg = linear_model.LinearRegression()
fit = reg.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
print(fit)
coef = reg.coef_
print(coef)
intercept = reg.intercept_
print(intercept)