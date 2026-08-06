def handle_nan_train(df) :

    expense_column = ["RoomService"	,
                    "FoodCourt" ,
                    "ShoppingMall" ,
                    "Spa" ,
                    "VRDeck"
    ]

    for i in expense_column :
        df.loc[(df["CryoSleep"] == True) & (df[i].isnull()) , i] = 0.0
    
    df.loc[(((df["HomePlanet"] == "Europa") | (df["HomePlanet"] == "Mars")) & (df["VIP"].isnull())) , "VIP"] = True
    df.loc[((df["HomePlanet"] == "Earth") & (df["VIP"].isnull())) , "VIP"] = False

    df.loc[((df["VIP"] == False) & (df["HomePlanet"].isnull())) , "HomePlanet"] = "Earth"

    df.loc[((df["Expense"] == 0.0) & (df["CryoSleep"].isnull())) , "CryoSleep"] = True
    df.loc[((df["Expense"] != 0.0) & (df["CryoSleep"].isnull())) , "CryoSleep"] = False

    for i in expense_column :
        df.loc[((df["VIP"] == True) & (df[i].isnull())) , i] = float((df[df["VIP"] == True])[i].median())

    for i in expense_column :
        df.loc[((df["VIP"] == False) & (df[i].isnull())) , i] = float((df[df["VIP"] == False])[i].median())

    unique_group = df["Group"].unique().tolist()

    for i in unique_group:
        cabin = df.loc[
            (df["Group"] == i) & (df["Cabin"].notnull()),
            "Cabin"
        ]

        if not cabin.empty:
            df.loc[
                (df["Group"] == i) & (df["Cabin"].isnull()),
                "Cabin"
            ] = cabin.iloc[0]

    df = df.dropna(subset = ["Cabin"])
    df = df.dropna(subset = ["HomePlanet"])

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Destination"] = df["Destination"].fillna(df["Destination"].mode()[0])

    return df


def handle_nan_test(df) :

    expense_column = ["RoomService"	,
                    "FoodCourt" ,
                    "ShoppingMall" ,
                    "Spa" ,
                    "VRDeck"
    ]

    for i in expense_column :
        df.loc[(df["CryoSleep"] == True) & (df[i].isnull()) , i] = 0.0
    
    df.loc[(((df["HomePlanet"] == "Europa") | (df["HomePlanet"] == "Mars")) & (df["VIP"].isnull())) , "VIP"] = True
    df.loc[((df["HomePlanet"] == "Earth") & (df["VIP"].isnull())) , "VIP"] = False

    df.loc[((df["VIP"] == False) & (df["HomePlanet"].isnull())) , "HomePlanet"] = "Earth"

    df.loc[((df["Expense"] == 0.0) & (df["CryoSleep"].isnull())) , "CryoSleep"] = True
    df.loc[((df["Expense"] != 0.0) & (df["CryoSleep"].isnull())) , "CryoSleep"] = False

    for i in expense_column :
        df.loc[((df["VIP"] == True) & (df[i].isnull())) , i] = float((df[df["VIP"] == True])[i].median())

    for i in expense_column :
        df.loc[((df["VIP"] == False) & (df[i].isnull())) , i] = float((df[df["VIP"] == False])[i].median())

    unique_group = df["Group"].unique().tolist()

    for i in unique_group:
        cabin = df.loc[
            (df["Group"] == i) & (df["Cabin"].notnull()),
            "Cabin"
        ]

        if not cabin.empty:
            df.loc[
                (df["Group"] == i) & (df["Cabin"].isnull()),
                "Cabin"
            ] = cabin.iloc[0]

    df["Cabin"] = df["Cabin"].fillna(df["Cabin"].mode()[0])
    df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode()[0])
    df["VIP"] = df["VIP"].fillna("False")

    df["Age"] = df["Age"].fillna(df["Age"].median())

    df["Destination"] = df["Destination"].fillna(df["Destination"].mode()[0])

    return df




def get_X_y(df) :

    df[["Group" , "pass_no"]] = df["PassengerId"].str.split("_" , expand = True)

    df["Expense"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]

    df = handle_nan_train(df)

    df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)

    df = df.drop(["Name" , "PassengerId" , "Cabin" , "pass_no" , "Expense"] , axis = 1)

    df["CabinNum"] = df["CabinNum"].values.astype(int)
    df["Group"] = df["Group"].values.astype(int)

    X = df.drop(["Transported"] , axis = 1)
    y = df["Transported"]

    y = y.astype(int)

    return X , y


def get_X(df) :

    df[["Group" , "pass_no"]] = df["PassengerId"].str.split("_" , expand = True)

    df["Expense"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]

    df = handle_nan_test(df)

    df[["Deck", "CabinNum", "Side"]] = df["Cabin"].str.split("/", expand=True)

    df = df.drop(["Name" , "PassengerId" , "Cabin" , "pass_no" , "Expense"] , axis = 1)

    df["CabinNum"] = df["CabinNum"].values.astype(int)
    df["Group"] = df["Group"].values.astype(int)

    return df