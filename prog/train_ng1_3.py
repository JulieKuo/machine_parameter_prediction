import pickle, os
import pandas as pd
from tensorflow.keras import models, layers, initializers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']



shuffle = True
random_state = True



def ng1_3(dfs, model_path, model_detail):
    # train model
    total_model = {}
    for i, df in enumerate(dfs, start = 1):
        print(f"model{i}".center(130, '_'))

        # split data
        X = df.drop("target", axis = 1)
        y = df["target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state, stratify = y, shuffle = shuffle)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.15, random_state = random_state, stratify = y_train, shuffle = shuffle)
        
        # save features    
        pickle.dump(X_train.columns, open(os.path.join(model_detail, f"features{i}.pkl"),  "wb"))

        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)    
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)
        
        # build model
        model = models.Sequential([
            layers.Dense(64, input_shape = (X.shape[1], ), kernel_initializer = initializers.he_normal()), # 使用He初始化
            layers.BatchNormalization(),
            layers.Activation("relu"),
            # layers.Dropout(0.1),
            layers.Dense(32, kernel_initializer = initializers.he_normal()), # 使用He初始化
            layers.BatchNormalization(),
            layers.Activation('relu'),
            # layers.Dropout(0.1),
            layers.Dense(3, kernel_initializer = initializers.glorot_normal()), # 使用Xavier初始化
            layers.Activation("softmax")
        ], name = f"model{i}")

        model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

        # train model
        early_stopping = callbacks.EarlyStopping(monitor = "val_loss", patience = 5)
        history = model.fit(X_train, y_train, epochs = 50, batch_size = 256, validation_data = (X_valid, y_valid), callbacks = [early_stopping])

        # save model
        pickle.dump(scaler, open(os.path.join(model_detail, f"scaler{i}.pkl"), "wb"))
        pickle.dump(model, open(os.path.join(model_detail, f"model{i}.pkl"),  "wb"))

        # save result
        y_train_pred = model.predict(X_train).argmax(axis = 1)
        y_valid_pred = model.predict(X_valid).argmax(axis = 1)
        y_test_pred = model.predict(X_test).argmax(axis = 1)

        total_model[i] = {
            "model": model,
            "training_loss": history.history['loss'],
            "validation_loss": history.history['val_loss'],
            "train":{
                "target": y_train.to_numpy(),
                "pred": y_train_pred
            },
            "valid":{
                "target": y_valid.to_numpy(),
                "pred": y_valid_pred
            },
            "test":{
                "target": y_test.to_numpy(),
                "pred": y_test_pred
            }
        }

    
    # calculate model score
    score = []
    for (key, values) in total_model.items():
        f1_train = f1_score(values["train"]["target"], values["train"]["pred"], average = 'macro')
        f1_valid = f1_score(values["valid"]["target"], values["valid"]["pred"], average = 'macro')
        f1_test = f1_score(values["test"]["target"], values["test"]["pred"], average = 'macro')
        score.append([key, f1_train, f1_valid, f1_test])
    score = pd.DataFrame(score, columns = ["model", "train", "valid", "test"]).round(4)
    score.to_csv(f"{model_path}/f1_score.csv", index = False)


    # plot loss curve
    for i, (key, values) in enumerate(total_model.items()):
        plt.figure(figsize = (5, 3))
        plt.plot(values["training_loss"], label="Training Loss")
        plt.plot(values["validation_loss"], label="Validation Loss")
        plt.title(f"model{key}")
        plt.legend()
        plt.savefig(f"{model_path}/loss_curve{key}.png")


    # plot confusion matrix
    for i, (key, values) in enumerate(total_model.items()):
        fig, ax = plt.subplots(1, 3, figsize = (15, 4))
        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(values["train"]["target"], values["train"]["pred"]))
        disp.plot(cmap = plt.cm.Blues, ax = ax[0])
        ax[0].set_title("train")

        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(values["valid"]["target"], values["valid"]["pred"]))
        disp.plot(cmap = plt.cm.Blues, ax = ax[1])
        ax[1].set_title("valid")

        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(values["test"]["target"], values["test"]["pred"]))
        disp.plot(cmap = plt.cm.Blues, ax = ax[2])
        ax[2].set_title("test")

        fig.suptitle(f"model{key}")
        fig.savefig(f"{model_path}/confusion_matrix{i+1}.png")


    return score