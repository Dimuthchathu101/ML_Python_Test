from sklearn import datasets
from sklearn.model_selection import train_test_split

iris_dataset = datasets.load_iris()

features = iris_dataset.data
labels = iris_dataset.target

print(features)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.5)
print(len(features_train))