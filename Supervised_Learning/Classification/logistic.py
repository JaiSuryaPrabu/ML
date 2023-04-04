# Import the dataset
import pandas as pd

data = pd.read_csv("ads.csv")
print(data.head())

# Splitting into the dependent and independent variables and testing and training data
from sklearn.model_selection import train_test_split
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

# Training the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(x_train,y_train)
pred = model.predict(x_test)

# Evaluation of the model
from sklearn.metrics import accuracy_score,confusion_matrix
score = accuracy_score(y_test,pred)
matrix = confusion_matrix(y_test,pred)
print("Accuracy Score : ",score)
print("Confusion matrix : \n",matrix)