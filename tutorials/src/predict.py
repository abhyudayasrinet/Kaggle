import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import dispatcher
import joblib

TRAINING_DATA = 'input/train_folds.csv'
TEST_DATA = 'input/test.csv'
MODEL = 'randomforest'
FOLD = 0

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df['id'].values
    predictions = None

    for FOLD in range(1):
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(f"models/{MODEL}_{FOLD}_label_encoder.pkl")    
        cols = joblib.load(f"models/{MODEL}_{FOLD}_columns.pkl")  
        for c in cols:
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        clf = joblib.load(f"models/{MODEL}_{FOLD}.pkl")
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    sub['id'] = sub['id'].astype(int)

    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)