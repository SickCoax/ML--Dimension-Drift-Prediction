from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from preprocessing import get_X_y
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def classifier(df) :

    X , y = get_X_y(df)

    X_train , X_val , y_train , y_val = train_test_split(
        X ,
        y ,
        test_size = 0.1 ,
        random_state = 7
    )

    cat_cols = X_train.select_dtypes(include = ["object" , "string"]).columns

    preprocess = ColumnTransformer([
        ("cat" , OneHotEncoder(handle_unknown= "ignore") , cat_cols)
    ] , remainder = "passthrough"
    )

    model = Pipeline([
        ("preprcoess" , preprocess) ,
        ("xgbc" , XGBClassifier(
            n_jobs = -1 ,
            subsample = 0.8 ,
            colsample_bytree = 0.8 ,
            random_state = 9 ,
            max_depth = 7 ,
            min_child_weight = 3 ,
            n_estimators = 325 ,
            gamma = 0.2624 ,
            learning_rate = 0.0455 ,
            reg_lambda = 4.5954 ,
            reg_alpha = 2.2881
        ))
    ])

    model.fit(X_train , y_train)

    return model

    # print(f"F1 Score : {f1}")
    # print(f"Accuracy Score : {ac}")