import os
import pandas as pd
from preprocessing import get_X
from train import classifier

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "train.csv"
)
df_train = pd.read_csv(csv_path)

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "test.csv"
)
df_test = pd.read_csv(csv_path)

X_test = get_X(df_test)

model = classifier(df_train)

y_pred = model.predict(X_test)

result = pd.DataFrame({
    "PassengerId" : df_test["PassengerId"] ,
    "Transported" : y_pred.astype(bool)
})

result.to_csv('result.csv', index=False)

print(result)