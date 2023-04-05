# Import the dataset
import pandas as pd

print("Data is imported ...")
data = pd.read_csv("ads.csv")
print("Data in the dataset : ")
print(data.head())

# Splitting into the dependent and independent variables and testing and training data
from sklearn.model_selection import train_test_split
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
print("Data are splitted into dependent and independent values ...")
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
print("Data are splitted into training and testing data ...")

# feature scaling
print("Features are ready to be scaled down ...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
print("Features are scaled ...")

# training model
print("Training the model ...")
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train,y_train)
print("SVC model is trained ...")
pred = classifier.predict(x_test)

# Evaluation
print("Evaluation of the model ...")
from sklearn.metrics import accuracy_score,confusion_matrix
score = accuracy_score(y_test,pred)
matrix = confusion_matrix(y_test,pred)
print(f"Accuracy Score : {score} \n Confusion matrix : \n{matrix}")

'''
Accuracy Score : 0.87      
Confusion matrix : 
[ [59 10]
  [ 3 28] ]
'''