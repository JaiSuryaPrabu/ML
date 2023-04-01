# Random Forest Regression

# Import the dataset
print("Importing the dataset\n")
import pandas as pd
data = pd.read_csv("house_price.csv")
print("The dataset is loaded and the first 5 rows ...")
print(data.head())

# Split the dependent and independent variable
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
print("\nSplitting the dataset into independent varialbles and dependent variables\n")

# Split the training and testing dataset
print("\nSplitting the dataset into training and testing and the shapes are:\n")
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y)
print(f"X training shape : {x_train.shape} \nY training shape : {y_train.shape} \n X testing shape : {x_test.shape} \nY testing shape : {y_test.shape}\n")

# Feature Scaling
print("Scaling the depenedent variable values ....\n")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Training
print("Training the model ...\n")
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=20)
model.fit(x_train,y_train)
print("The model is trained...\n")

# Testing
print("Testing process begins....\n")
from sklearn.metrics import r2_score
y_pred = model.predict(x_test)
score = r2_score(y_test,y_pred)
print("The score of random forest regressor is : ",score)
