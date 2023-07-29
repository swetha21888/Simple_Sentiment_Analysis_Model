from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

# Training data
positive_set = ["we are lucky", "we are loved", "we will win", "we are best", "we are happy"]
negative_set = ["we are hated", "we are unlucky", "we are sad", "we are worst", "we will lose"]

# Sample data for prediction
sample_set = ["we are lucky", "we are unlucky", "we are worst", "we are happy", "we are sad"]

# Combine positive and negative sets to create the complete dataset
data_set = positive_set + negative_set

# Create labels for the data_set
data_labels = ["POSITIVE"] * len(positive_set) + ["NEGATIVE"] * len(negative_set)

# Create a CountVectorizer object and transform the data into numerical feature vectors
vectorizer = CountVectorizer()
vectorizer.fit(data_set)
data_vectors = vectorizer.transform(data_set)
sample_vectors = vectorizer.transform(sample_set)

# Create a DecisionTreeClassifier and train it on the data_vectors and data_labels
classifier = tree.DecisionTreeClassifier()
classifier.fit(data_vectors, data_labels)

# Make predictions on the sample_vectors using the trained classifier
predictions = classifier.predict(sample_vectors)

# Print the predictions
print(predictions)
