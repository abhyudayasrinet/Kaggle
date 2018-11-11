import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


def data_cleanup(df, drop_columns):
    # clean up data
    df['Age'] = df['Age'].fillna(-0.5)
    df['Cabin'] = df['Cabin'].fillna('N')
    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

    #drop the unwanted columnss
    for column in drop_columns:
        print('Dropping '+column)
        df = df.drop(column, axis = 1)

    #bin the ages
    bins = [-1, 0, 6, 13, 18, 25, 35, 50, 120]
    labels = ["Unknown", "Baby", "Child", "Teenager", "Young Adult", "Adult", "Old", "Senior"]
    df['Age'] = pd.cut(df['Age'], bins = bins, labels = labels)

    #bin the fare
    fare_stats = df['Fare'].describe()
    bins = [-1, fare_stats['min'], fare_stats['25%'], fare_stats['50%'], fare_stats['75%'], fare_stats['max']+100]
    labels = ["Unknown", "1st_quartile", "2nd_quartile", "3rd_quartile", "4th_quartile"]
    df['Fare'] = pd.cut(df['Fare'], bins = bins, labels = labels)

    return df

def encode_features(df_train, df_test):
    features = ['Fare', 'Age', 'Sex']
    df_combined = pd.concat([df_train[features], df_test[features]])
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test



def logistic_regression(train, test):
    print("Logistic Regression...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    print(model.score(X_train, Y_train)) #Accuracy score for training set


    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("Accuracy", metrics.accuracy_score(Y_test, predicted)) #Accuracy score for test set

    #Prediction on Cross Validation
    scores = cross_val_score(LogisticRegression(), X, Y, scoring='accuracy', cv=10)
    print(scores)
    print(scores.mean())

    print(test.head())
    passenger_ids = test['PassengerId']
    predictions = model.predict(test.drop(columns=["PassengerId"]))
    output = pd.DataFrame({'PassengerId': passenger_ids, "Survived" : predictions})
    output.to_csv("output.csv", index=False)


def data_explore(train):
    
    # train = train.drop("PassengerId", axis = 1)

    for feature in train.columns:
        if(feature in ["Survived", "Name", "Age"]):
            continue
        print(train[[feature, "Survived"]].groupby(feature, as_index=False).mean())
        print(train[feature].value_counts())
        pd.crosstab(train[feature], train["Survived"]).plot(kind='bar')
        plt.xlabel(feature)
        plt.ylabel("Survived")
        plt.show()        
        

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# data_explore(train)

train = data_cleanup(train, ['Name', 'Embarked', 'Ticket', 'Cabin', 'PassengerId'])
test = data_cleanup(test, ['Name', 'Embarked', 'Cabin', 'Ticket'])

data_explore(train)

Y = train.iloc[:,0]
X = train.iloc[:,1:]

print(X.head())
print(test.head())


train, test = encode_features(X, test)

logistic_regression(train, test)


