import pandas as pd
data = pd.read_csv("classificate_model - Copy.csv")

#print(data)

from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split

#split the dataset intor training and validation datasets
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, data.Classification, test_size=0.2, random_state=42,stratify=data["Classification"])
#print(x_train)
#print(y_test)
#print("x_train.head() -------------------------\n", x_train.head())
#print("y_train.head() -------------------------\n", y_train.head())
#print("x_test.head() -------------------------\n", x_test.head())
#print("y_test.head() -------------------------\n",y_test.head())
#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer

##TfidfVectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2,3))
TfidfVectorizer = TfidfVectorizer()
x_train = TfidfVectorizer.fit_transform(x_train.Words)
x_test = TfidfVectorizer.fit_transform(x_test.Words)

#print("x_train.shape -------------------------\n",x_train.shape)
#print("x_test.shape -------------------------\n",x_test.shape)
#print("x_train-------------------------\n",x_train)
#print("x_test -------------------------\n",x_test)

##----------------------------------------------Text Classification [Naive Bayes Classifier]----------------------------------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

classifer = MultinomialNB()
alpha_ranges = {"alpha":[10**-2,10**-1,10**0,10**1,10**2]} #find out which is the best alpha
grid_search = GridSearchCV(classifer, param_grid = alpha_ranges, scoring="accuracy", cv=3,return_train_score=True) #find out what cv=3 means
grid_search.fit(x_train,y_train)
#print("grid_search.fit(x_train,y_train)", grid_search.fit(x_train,y_train))

alpha = [10**-2,10**-1,10**0,10**1,10**2]
train_accuracy = grid_search.cv_results_['mean_train_score']
train_std = grid_search.cv_results_['std_train_score']
test_accuracy = grid_search.cv_results_['mean_test_score']
test_std = grid_search.cv_results_['std_test_score' ]

'''import matplotlib.pyplot as plt
#how to get validation curve https://www.geeksforgeeks.org/validation-curve/
# Plot mean accuracy scores for training and testing scores
plt.plot(alpha, train_accuracy,
     label = "Training Score", color = 'b')
plt.plot(alpha, test_accuracy,
   label = "Cross Validation Score", color = 'g')
 
# Creating the plot
plt.title("Validation Curve with Naive Bayes Classifier")
plt.xlabel("Alpha_range")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.legend(loc = 'best')
plt.show()'''


##---------------------------------------------- [Naive Bayes Classifier] prediction----------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
le_train = LabelEncoder()
y_train=le_train.fit_transform(y_train)
print("y_train", y_train)

le_test= LabelEncoder()
y_test=le_test.fit_transform(y_test)
print("y_test", y_test)

#print("grid_search.best_estimator_", grid_search.best_estimator_)
mNB = MultinomialNB(alpha=0.1) #classifier based on print("grid_search.best_estimator_", grid_search.best_estimator_) = (alpha=0.1)
predict = mNB.fit(x_train, y_train).predict(x_test)
#print("accuracy is : ", accuracy_score(valid_y, predict))




