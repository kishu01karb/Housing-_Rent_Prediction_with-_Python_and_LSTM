# Problem Statement:
#Write a Python program that:
#Loads and explores a housing rent dataset.
#import libraries
#First question is what type of libraries will i need to import

import pandas as pd  #for  reading and manipulating data
import numpy as np   #For mathematical operations
import matplotlib.pyplot as plt # for plotting data
import  plotly.express as px # also for plotting quick and easy charts
import plotly.graph_objects as go # for detailed and customized charts

data = pd.read_csv('House_Rent_Dataset.csv')
print(data.head()) #to display top 5 data
#Preprocesses the data by handling categorical and numerical features.
#now we will check if data contains null values or not
print(data.isnull().sum())
#descriptive stat to get overview of how data is looking now and check outliers or errors
print(data.describe())
#now we will calculate min ,median ,highest , lowest rent to understand the full picutre of data
print(f"Mean of rent is {data.Rent.mean()}") # gives average of all rents and general idea of overall cost
print(f"Median of rent is {data.Rent.median()}") # the rent value in the middle shows true center of data
print(f"Highest rent is {data.Rent.max()} ") # the most expensive rents helps identify range and unrealistic values
print(f"Lowest rent is {data.Rent.min()}") # the least expensive one
#chart of how rent of the houses in different cities lloks like acc to no of bedrooms , hallsand kitchens
figure = px.bar(data, x = data["City"], y= data["Rent"], color =data["BHK"], title ="Rent in different Cities")
figure.show()
#what do we learn from it 1.which city has affordable 2bhks,2. where rent rises steeeply,3. cities where larger houses are better value,4.helps users make informed choices for living or investment
#Visualizes patterns and relationships between rent and influencing features.
#lets look at remt of houses in different cities according to area type
figure1 = px.bar(data, x=data["City"],y=data["Rent"], color = data["Area Type"], title ="Rent in different cities according to area type ")
figure1.show()
#lets look at rent of the houses in different cities acc to furnishing status
figure2 = px.bar(data, x= data["City"], y=data["Rent"], color = data["Furnishing Status"], title = "Rent in different citites according to furnished status")
figure2.show()
#lets look at rent of the houses of different cities acc to size
figure3 = px.bar(data, x= data["City"], y=data["Rent"], color = data["Size"], title =" Rent in different cities according to size of houses")
figure3.show()
#lets look at houses available fpr rent in dofferent cities
cities = data["City"].value_counts()
label = cities.index
counts = cities.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts, hole =0.5)])
fig.update_layout(title_text='Number of houses available for rent')
fig.update_traces(hoverinfo ='label+percent', textinfo='value', textfont_size = 30, marker=dict(colors=colors , line = dict(color='black', width=3)))
fig.show()

#no of houses available for  different types of tenants
#prefernce of tenant
tenant = data["Tenant Preferred"].value_counts()
label = tenant.index
counts = tenant.values
colors = ['gold','lightgreen']
fig1 = go.Figure(data=[go.Pie(labels=label, values=counts, hole=0.5)])
fig1.update_layout(title_text='Preference of Tenant in India')
fig1.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig1.show()

#Builds a Machine Learning model using LSTM to predict rent based on input features.
#we are gonna transform features that is independent variable which target varaible depend on here rent into numeric vallues so computer can process and identify easily
data["Area Type"] = data["Area Type"].map({"Super Area":1,"Carpet Area":2,"Built Area":3})
data["City"] = data["City"].map({"Mumbai": 4000,"Chennai":6000,"Bangalore":5600,"Hyderabad":5000,"Delhi":1100,"Kolkata": 7000})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished":0,"Semi-Furnished":1,"Furnished":2})
data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors/Family":2,"Bachelors":1,"Family":3})
print(data.head())

#now for teaching machine to predict we will split data into trainig data which it will learn on and testing data which it will check how much it has learnt is correct
from  sklearn.model_selection import train_test_split
#creating input and output arrays that is input features x and target values y to learn patterns
x = np.array(data[["BHK", "Size", "Area Type", "City",
                   "Furnishing Status", "Tenant Preferred",
                   "Bathroom"]])
y = np.array(data[["Rent"]])
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.10,random_state =42)
#now we will built lstm layers to learn sequence ot pattern in housing daata and predict rent based on it

from keras.models import Sequential
from keras.layers import Dense, LSTM

model =Sequential()# to create blank model where layers can be added in sequence like container for our nrueral network layers allow us to define the model step by step from input to output
model.add(LSTM(128, return_sequences =True, input_shape =(xtrain.shape[1],1)))
#this Lstm layer is the first layer and needs the shape of the input to process the data
model.add(LSTM(64,return_sequences=False))
#processes th op from the previous lstm layer and summarises the seq to a single vector that is compresses the seq for next fully connecred layers
model.add(Dense(25))
#it introduces complexity or non linearity to compreesed seq vector to learn patternss and each nuerons applies transformation to input using learned weights
model.add(Dense(1))
#to  produce final predicted rent value which is single continuous number
model.summary()
#to verify and inspect the structure
model.compile(optimizer='adam', loss='mae')
#prepares the model for training tells model how to learn loss helps us manage reduce the error
model.fit(xtrain,ytrain, batch_size =1, epochs = 50)
# Calculate metrics
# Predict on test data
#trains model usng training data tells model what data to learn from and how long
#“Hey! Learn to predict rent by looking at this data. Use Adam to help you learn smartly, and try to make as few mistakes as possible (using mean squared error). Go over the whole dataset 21 times, one example at a time.”

#Accepts user input and predicts house rent using the trained model.
print("enter house details to predict rent")
a = int(input("Number of BHK: "))
b = int(input("Size of the House: "))
c = int(input("Area Type (Super Area = 1, Carpet Area = 2, Built Area = 3): "))
d = int(input("Pin Code of the City: "))
e = int(input("Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): "))
f = int(input("Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): "))
g = int(input("Number of bathrooms: "))
features = np.array([[a, b, c, d, e, f, g]])
print("Predicted House Price = ", model.predict(features))