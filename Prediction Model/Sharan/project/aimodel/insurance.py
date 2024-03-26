#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import locale


# In[100]:


locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


# In[101]:


data = pd.read_csv("./aimodel/Train_Data.csv")


# In[102]:


le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])


# In[103]:


X = data[['age', 'sex', 'bmi', 'smoker']]
y = data['charges']


# In[104]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[105]:


y_pred = model.predict(X_test)
r_squared = r2_score(y_test, y_pred)
print("R-squared:", r_squared)


# In[106]:


joblib.dump(model, "insurance_model.pkl")


# In[107]:


def predict_charges(age, sex, bmi, smoker):
    model = joblib.load("insurance_model.pkl")
    input_data = [[age, sex, bmi, smoker]]
    predicted_charges = model.predict(input_data)
    return predicted_charges[0]


# In[109]:


# age = float(input("Enter age: "))
# sex = int(input("Enter sex (0 for female, 1 for male): "))
# bmi = float(input("Enter BMI: "))
# smoker = int(input("Enter smoker status (0 for non-smoker, 1 for smoker): "))
# predicted_charges = predict_charges(age, sex, bmi, smoker)
# print("Predicted charges:", predicted_charges)


# In[ ]:




