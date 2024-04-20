import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import urllib.request

plt.style.use('seaborn')
matplotlib.use('Agg')

print('Downloading Updated Dataset')
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
urllib.request.urlretrieve(url, 'timeSeriesDataset/time_series_covid19_confirmed_global.csv')
print('Successfully Downloaded')

def getData(country, days_in_future):
    days_in_future = int(days_in_future)
    confirmed_cases = pd.read_csv('timeSeriesDataset/time_series_covid19_confirmed_global.csv')
    cols = confirmed_cases.keys()
    confirmed = confirmed_cases.loc[:, cols[4]:cols[-1]]
    dates = confirmed.keys()
    world_cases = []

    if country == "World":
        for i in dates:
            confirmed_sum = confirmed[i].sum()
            world_cases.append(confirmed_sum)
    else:
        country_index = confirmed_cases['Country/Region'].loc[confirmed_cases['Country/Region'] == country].index[0]
        for i in dates:
            confirmed_sum = confirmed[i][country_index]
            world_cases.append(confirmed_sum)

    days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
    all_cases = np.array(world_cases).reshape(-1, 1)

    future_forecast = np.array([i for i in range(len(dates) + days_in_future)]).reshape(-1, 1)
    adjusted_dates = future_forecast[:-days_in_future]

    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forecast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

    return days_since_1_22, all_cases, start_date, future_forcast_dates, adjusted_dates, future_forecast, days_in_future


def makePrediction(algorithm, days_since_1_22, all_cases, start_date, future_forcast_dates, adjusted_dates,
                   future_forecast, days_in_future, savePath, country):
    X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22,
                                                                                                all_cases,
                                                                                                test_size=0.1,
                                                                                                shuffle=False)
    if algorithm == "SVM":
        print('SVM')
        kernel = ['poly']
        c = [0.01]
        gamma = [0.01]
        epsilon = [0.01]
        shrinking = [False]
        svm_grid = {'kernel': kernel, 'C': c, 'gamma': gamma, 'epsilon': epsilon, 'shrinking': shrinking}
        svm = SVR()
        svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True,
                                        n_jobs=-1, n_iter=100, verbose=1)
        svm_search.fit(X_train_confirmed, y_train_confirmed)
        svm_confirmed = svm_search.best_estimator_
        pred = svm_confirmed.predict(future_forecast)
        test_pred = svm_confirmed.predict(X_test_confirmed)

    elif algorithm == "Linear Regression":
        print("LinearRegression")
        linear_model = LinearRegression(normalize=True, fit_intercept=True)
        linear_model.fit(X_train_confirmed, y_train_confirmed)
        pred1 = linear_model.predict(future_forecast)
        test_pred1 = linear_model.predict(X_test_confirmed)
        pred = []
        test_pred = []
        for it in pred1:
            pred.append(it[0])
        for it in test_pred1:
            test_pred.append(it[0])

    elif algorithm == "Decision Tree":
        print("Decision Tree")
        decisionClassifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
        decisionClassifier.fit(X_train_confirmed, y_train_confirmed)
        pred = decisionClassifier.predict(future_forecast)
        test_pred = decisionClassifier.predict(X_test_confirmed)

    elif algorithm == "Random Forest":
        print("Random Forest")
        randomClassifier = RandomForestClassifier()
        randomClassifier.fit(X_train_confirmed, y_train_confirmed)
        pred = randomClassifier.predict(future_forecast)
        test_pred = randomClassifier.predict(X_test_confirmed)

    print(test_pred)
    start = '1/22/2020'
    start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
    future_forcast_dates = []
    for i in range(len(future_forecast)):
        future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

    # print(future_forcast_dates)
    # print(len(future_forcast_dates))
    # print(len(pred))

    future_cases = set(zip(future_forcast_dates[-days_in_future:], pred[-days_in_future:]))

    futureDates = []
    futureCases = []
    data = list(future_cases)
    data.sort(key=lambda x: x[1])
    for i in range(len(data)):
        data[i] = (data[i][0], int(data[i][1]))
        futureCases.append(int(data[i][1]))
        futureDates.append(data[i][0])

    print('MAE:', mean_absolute_error(test_pred, y_test_confirmed))
    print('MSE:', mean_squared_error(test_pred, y_test_confirmed))

    plt.figure(figsize=(20, 12))
    plt.plot(adjusted_dates, all_cases)
    plt.plot(future_forecast, pred, linestyle='dashed', color='purple')
    plt.title('Number of Coronavirus Cases Over Time (' + country + ')', size=30)
    plt.xlabel('Days Since 1/22/2020', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.legend(['Confirmed Cases', algorithm + ' predictions'], fontsize='xx-large')
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.savefig(savePath + "\\templates\\assets\\images\\futureCases.png")
    plt.close()

    plt.figure(figsize=(25, 15))
    plt.plot(futureDates, futureCases, 'o-g')
    plt.title('Number of Coronavirus Cases Over Time (' + country + ')', size=30)
    plt.xlabel('Predicted Days Cases', size=30)
    plt.ylabel('Number of Cases', size=30)
    plt.legend(['Confirmed Cases', algorithm + ' predictions'], fontsize='xx-large')
    plt.xticks(size=30)
    plt.yticks(size=15)
    plt.xticks(rotation=80)
    plt.savefig(savePath + "\\templates\\assets\\images\\futureDatesCases.png")
    plt.close()

    return True, future_cases
