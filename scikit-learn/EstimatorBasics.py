from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

# Rows = samples, Columns = features
X = [[ 1,  2,  3], 
     [11, 12, 13]]

# Classes the training data is sorted into in supervised learning. 
# Obviously not needed for unsupervised learning

y = [1, 2]

clf.fit(X, y)

print(clf.predict([[4, 5, 6], [14, 15, 16]]))