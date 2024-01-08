from django.shortcuts import render;

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")

def result(request):
    data = pd.read_csv(r'D:\Machine-learning\Data\USA_Housing.csv')

    data = data.drop(['Address'], axis=1)

    X = data.drop(['Price'], axis=1)  # droped column Price
    y = data['Price']  # take column Price

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_data = X_train.join(y_train)
    test_data = X_test.join(y_test)

    X_train, y_train = train_data.drop(['Price'], axis=1), train_data['Price']
    X_test, y_test = test_data.drop(['Price'], axis=1), test_data['Price']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    reg = LinearRegression()
    reg.fit(X_train, y_train)

    sgdr = SGDRegressor(max_iter=100000)
    sgdr.fit(X_train, y_train)

    b_norm = sgdr.intercept_
    w_norm = sgdr.coef_

    regr = RandomForestRegressor(max_depth=15, random_state=0)
    regr.fit(X_train, y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])

    input_s = scaler.fit_transform(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
    input = np.array([var1, var2, var3, var4, var5]).reshape(1, -1)
    # prediction on Training data
    y_pred = regr.predict(input)

    # y_pred = np.dot(X_train_scaled, w_norm) + b_norm
    prediction = round(y_pred[0])
    price = "The prediction price is $"+str(prediction)
    return render(request, "predict.html", {"result2":price })