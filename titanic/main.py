import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#%matplotlib inline


def simplify_ages(df):
    df.Age = df.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.Age, bins, labels=group_names)
    df.Age = categories
    return df

def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df

# def formatPrefix(x):

    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    # print(df_train[features].head())
    df_combined = pd.concat([df_train[features], df_test[features]])
    # print(df_combined.head())
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test


data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

pd.set_option('display.height', 5000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 5000)

# print(data_train.describe().to_string())
# print(data_train.sample(3))
data_train = simplify_ages(data_train)
sns.barplot(x="Age", y="Survived", data=data_train)
plt.show()

# sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train, palette={"male": "blue", "female": "pink"}, markers=["*", "o"], linestyles=["-", "--"])
# plt.show()

# print(data_train.Name.describe())


# print(data_train.head())
# print(data_train["NamePrefix"].unique())
# print(data_train.NamePrefix.unique())
# print(data_train.NamePrefix.apply(lambda x : x[0]))
# sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train)
# plt.show()

def machine_learning(data_train, data_test):

    data_train = transform_features(data_train)
    data_test = transform_features(data_test)

    data_train, data_test = encode_features(data_train, data_test)
    # print(data_train.head())

    X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
    y_all = data_train['Survived']

    #Split the data
    num_test = 0.20 #80-20 split
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

    #Train the model
    model = LogisticRegression()
    model = model.fit(X_train, y_train)
    print(model.score(X_train, y_train)) #Accuracy score for training set

    #Make predictions
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print(metrics.accuracy_score(y_test, predicted)) #Accuracy score for test set

    #Prediction on Cross Validation
    scores = cross_val_score(LogisticRegression(), X_all, y_all, scoring='accuracy', cv=10)
    print(scores)
    print(scores.mean())

    passenger_ids = data_test["PassengerId"]
    predictions = model.predict(data_test.drop(columns=["PassengerId"]))
    output = pd.DataFrame({'PassengerId':passenger_ids, "Survived": predictions})
    output.to_csv('predictions.csv',index=False)