from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


X = [[0, 15],
     [1, -10]]

clf = RandomForestClassifier(random_state=0)

# scale data according to computed scaling values
scaled = StandardScaler().fit(X).transform(X)
print(scaled)