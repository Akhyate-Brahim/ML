from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load and preprocess dataset
digits = datasets.load_digits()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(digits.data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data_scaled, digits.target, test_size=0.2, random_state=42)

# Train models
logistic_model = LogisticRegression(max_iter=10000, random_state=42)
logistic_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

kmeans_model = KMeans(n_clusters=10, random_state=42)
kmeans_model.fit(X_train)

# Save models
dump(logistic_model, 'logistic_model.joblib')
dump(tree_model, 'tree_model.joblib')
dump(kmeans_model, 'kmeans_model.joblib')

#%%
