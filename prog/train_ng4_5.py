import os, pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']


shuffle = True
random_state = True



def ng4_5(df, model_path, model_detail):
    # find ER samples
    ER_col = [i for i in df.columns if i[3] == "ER"]
    df = df[(df[ER_col] != 0).any(axis = 1)].reset_index(drop = True)
    
    
    # calculate total ER by each side
    left_ER_col  = [i for i in df.columns if (i[3] == "ER") and (i[1] == "左邊")]
    right_ER_col = [i for i in df.columns if (i[3] == "ER") and (i[1] == "右邊")]
    df["left"]  = df[left_ER_col].sum(axis = 1)
    df["right"] = df[right_ER_col].sum(axis = 1)


    # calculate difference between left and right
    error = abs(df["left"] - df["right"])
    error.value_counts(normalize = True)


    # get sample which can be adjusted
    non_index = df[error > 0.001].index # set a threshold (~70%)
    df = df.drop(non_index).reset_index(drop = True)


    # generate target
    df["target"] = (df["left"] >= df["right"]).astype(int)
    df["target"].value_counts()


    # remove constant value
    drop_flag = (df.nunique() == 1)
    keep_cols = df.columns[~drop_flag]
    df = df[keep_cols].iloc[:, 1:].drop(["left", "right"], axis = 1)


    # rename columns
    df.columns = [(cols[2] + "_" + cols[3]) if cols[0] != "target" else cols[0] for cols in df.columns]


    # split data
    X = df.drop("target", axis = 1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state, stratify = y, shuffle = shuffle)


    # train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    # save model
    pickle.dump(model, open(os.path.join(model_detail, f"model.pkl"),  "wb"))
    pickle.dump(X_train.columns, open(os.path.join(model_detail, f"features.pkl"),  "wb"))


    # evaluate cross validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv = 3, scoring = 'f1')
    cv_score = (cv_scores.mean())


    # calculate score
    f1_train = f1_score(y_train, y_train_pred, average = 'macro')
    f1_test = f1_score(y_test, y_test_pred, average = 'macro')
    score = pd.DataFrame([[f1_train, f1_test, cv_score]], columns = ["train", "test", "cv"]).round(4)
    score.to_csv(f"{model_path}/f1_score.csv", index = False)
    
    
    # evaluate performance
    fig, ax = plt.subplots(1, 2, figsize = (10, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_train, y_train_pred))
    disp.plot(cmap = plt.cm.Blues, ax = ax[0])
    ax[0].set_title("train")

    disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_test_pred))
    disp.plot(cmap = plt.cm.Blues, ax = ax[1])
    ax[1].set_title("test")
    
    fig.savefig(f"{model_path}/confusion_matrix.png")


    # feature importance
    fig, ax = plt.subplots(figsize = (15, 6))
    imp = pd.DataFrame(model.feature_importances_, index = X_train.columns, columns = ["importance"])
    imp = imp.query("importance != 0").sort_values("importance").head(10)
    imp.plot(kind = "barh", ax = ax, fontsize = 12)
    ax.set_title("Feature importances", fontsize = 14)
    plt.tight_layout()
    fig.savefig(f"{model_path}/importance.png")

    return score