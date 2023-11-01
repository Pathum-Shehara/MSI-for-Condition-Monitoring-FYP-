import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from matplotlib.colorbar import ColorbarBase

# Load the dataset from the CSV file
dataset = pd.read_csv('2.Palm Oil Transmittance Dark Current Reducted Bilateral Filtered Train Data Set (Super Pixel).csv',header = None)
test_dataset = pd.read_csv('2.Palm Oil Transmittance Dark Current Reducted Bilateral Filtered Test Data Set (Super Pixel).csv',header = None)

# Extract the feature vectors and labels
features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

test_features = test_dataset.iloc[:, :-1].values
test_labels = test_dataset.iloc[:, -1].values

#print(features.shape)
#print(labels)

# Create an instance of LDA
lda = LDA()

# Fit LDA to the dataset
lda.fit(features, labels)

# Transform the dataset to the LDA space
lda_features = lda.transform(features)
lda_testData = lda.transform(test_features)
#print(lda_features)
#print(lda_features.shape)

#The following code constructs the Scree plot
per_var = np.round(lda.explained_variance_ratio_* 100, decimals=1)
print(per_var)

acc = 99
cumulative = 0
lda_lim = 0
for i in per_var:
    cumulative += i
    lda_lim +=1
    if cumulative > acc:
        break
'''  
print(lda_lim)

labels = np.array(labels).astype(int).reshape(-1, 1)

data = np.hstack((lda_features[:, :lda_lim], labels))

df = pd.DataFrame(data , columns=['LDA' + str(x) for x in range(1, lda_lim+1)] + ['Labels'])

df.to_csv('Tumeric_Transmittance_HistogramEqualization_with_LDA_99%.csv', index=False)



# Split the transformed features and labels into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(lda_features, labels, test_size=0.2, random_state=42)

# Create and train the SVM classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
predictions = svm_classifier.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

# Assuming you have already applied LDA and obtained lda_features (transformed features)

# Define the categories and corresponding colors
categories = np.unique(labels)
num_categories = len(categories)
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']

# Create a dictionary to map each category to a color
color_mapping = dict(zip(categories, colors[:num_categories]))

# Create a scatter plot
scatter = plt.scatter(lda_features[:, 0], lda_features[:, 1], c=[color_mapping[label] for label in labels])
plt.xlabel('LDA Feature 1')
plt.ylabel('LDA Feature 2')
plt.title('LDA Scatter Plot')

# Create a legend with color categories outside the plot
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[label], markersize=8, label=label) for label in categories]
plt.legend(handles=legend_elements, bbox_to_anchor=(0.98, 1), loc='upper left', title='Categories')

plt.show()
'''
#print(lda_features.shape[0])
data = np.hstack((lda_features.reshape(lda_features.shape[0],lda_features.shape[1]), labels.reshape(lda_features.shape[0],1)))
df = pd.DataFrame(data, columns = ['LDA' + str(x) for x in range(1, lda_features.shape[1]+1)] + ['Labels'])
df.to_csv('6.Palm Oil Transmittance Dark Current Reducted Bilateral Filtered Train Data Set (Super Pixel) with LDA.csv', index=False)

test_data = np.hstack((lda_testData.reshape(lda_testData.shape[0],lda_testData.shape[1]), test_labels.reshape(lda_testData.shape[0],1)))
test_df = pd.DataFrame(test_data, columns = ['LDA' + str(x) for x in range(1, lda_testData.shape[1]+1)] + ['Labels'])
test_df.to_csv('6.Palm Oil Transmittance Dark Current Reducted Bilateral Filtered Test Data Set (Super Pixel) with LDA.csv', index=False)

