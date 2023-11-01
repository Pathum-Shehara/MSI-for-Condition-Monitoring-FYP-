import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

pixel_size = 10

trainData = pd.read_csv("6.Palm Oil Transmittance Dark Current Reducted Bilateral Filtered Train Data Set (Super Pixel) with LDA.csv")
testData = pd.read_csv("6.Palm Oil Transmittance Dark Current Reducted Bilateral Filtered Test Data Set (Super Pixel) with LDA.csv")
#print(Turmeric.head())
#print(Turmeric.describe())
#print(Turmeric.shape)

#Turmeric.plot(kind="scatter", x="0", y="4")
#sns.jointplot(x="2", y="4", data=Turmeric, size=5)

#sns.FacetGrid(Turmeric, hue="Out", size=4) \
   #.map(plt.scatter, "2", "8") \
   #.add_legend()

#sns.pairplot(Turmeric.drop("Id", axis=1), hue="Out", size=3)

#separating the data into dependent and independent variables
X_train = trainData.iloc[:, :-1].values
y_train = trainData.iloc[:, -1].values

X_test = testData.iloc[:, :-1].values
y_test = testData.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

'''
# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''
'''
# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''
'''
# Support Vector Machine's 
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''
'''
# Decision Tree's
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''
'''
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''
'''
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''
'''
# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''
'''
# Complement Naive Bayes
from sklearn.naive_bayes import ComplementNB
classifier = ComplementNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))
'''

from sklearn.metrics import accuracy_score, log_loss
#classifiers = [GaussianNB(),LogisticRegression(),DecisionTreeClassifier()]
classifiers = [GaussianNB(),BernoulliNB(),LogisticRegression(),KNeighborsClassifier(),SVC(),DecisionTreeClassifier()]
#classifiers = [GaussianNB(),MultinomialNB(),BernoulliNB(),ComplementNB(),LogisticRegression(),KNeighborsClassifier(),SVC(),DecisionTreeClassifier()] 
 
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
 
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    accuracy_sum = 0

    for i in range(0, testData.shape[0], 100):
        X_test = testData.iloc[i:i+100, :-1].values
        y_test = testData.iloc[i:i+100, -1].values
        y_pred = clf.predict(X_test)
        y_pred = y_pred.astype(int)
        y_pred = np.argmax(np.bincount(y_pred))
        print(y_pred)
        #print(y_test, y_pred)
        #print(accuracy_score(y_pred,y_test))
        #print(classification_report(y_test, y_pred))
        if y_pred == y_test[0]:
            accuracy_sum += 1
        else:
            accuracy_sum += 0
        #accuracy_sum += accuracy_score(y_pred,y_test)

    acc = accuracy_sum / (testData.shape[0]/(pixel_size**2))
    #train_predictions = clf.predict(X_test)
    #acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    log_entry = pd.DataFrame([[name, acc*100, 11]], columns=log_cols)
    log = pd.concat([log,log_entry], ignore_index = True)
    #log = log.append(log_entry)
    
    print("="*30)

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
ax = sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %',fontsize=18)
plt.ylabel('Classifier', fontsize=18)
plt.title('Classifier Accuracy for Palm Oil Transmittance Dark Current Reducted Bilateral Filtered (Test Train) Data Set (Super Pixel) with LDA',fontsize=20)

# Set the limits of the x-axis to 0 and 100
plt.xlim([0, 100])

# Add text annotations to the bars
# Add a text label to each bar
for i, v in enumerate(log['Accuracy']):
    ax.text(v + 1, i + .25, str("{:.2f}%".format(v)), color='black', fontweight='bold', fontsize=18)
    
# Increase the font size of the y-axis labels
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)

fig = plt.gcf()
fig.set_size_inches(26, 10)
plt.savefig('6.Classifier Accuracy for Palm Oil Transmittance Dark Current Reducted Bilateral Filtered (Test Train) Data Set (Super Pixel) with LDA.png', dpi=300)
plt.show()
