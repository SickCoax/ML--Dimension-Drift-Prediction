from sklearn.metrics import f1_score , accuracy_score

def get_scores(model , X_val , y_val) :

    y_pred = model.predict(X_val)

    f1 = f1_score(y_val , 
                  y_pred
        )
    
    ac = accuracy_score(y_val , 
                        y_pred
        )

    return f1 , ac