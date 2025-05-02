from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

data=pd.read_csv('Crop_recommendation.csv')

X=data[['N','P','K','temperature','humidity','ph','rainfall']]
y=data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)

prediction= model.predict(X_test)
accuracy=accuracy_score(y_test,prediction)
print("Accuracy:",accuracy)

new_features=pd.DataFrame([[90,  42,  43,    20.879744,  82.002744, 6.502985,  202.935536]])
predicted=model.predict(new_features)
print("Predicted Crop:",predicted)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved successfully!")
