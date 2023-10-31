import pandas as pd


def split_data(df_raw, data_split, model_cols):
    # split data by model
    dfs = []
    for model, values in model_cols.items():
        cols = values["secondary"] + values["primary"]
        cols = [i for i in df_raw.columns if (i[2] in cols)]
        dfs.append(df_raw[cols])


    # deal with constant value
    constant_summary = []
    for i in range(len(dfs)):
        drop_flag = (dfs[i].nunique() == 1)
        keep_cols = dfs[i].columns[~drop_flag]
        constant_summary.append({
            "init":  dfs[i].shape[1], 
            "drop":  drop_flag.sum(), 
            "keep":  len(keep_cols)
            })
        dfs[i] = dfs[i][keep_cols]
    constant_summary = pd.DataFrame(constant_summary, index = ["model1", "model2", "model3", "model4"])
    
    
    # generate target
    target_counts = pd.DataFrame()
    target_summary = []
    for i, df in enumerate(dfs):
        DV_col = [i for i in df.columns if i[3] == "DV"]
        DV_sum = df[DV_col[:-1]].sum(axis = 1).round(4)
        lower = (DV_sum.median() - DV_sum.std()/2).round(4)
        upper = (DV_sum.median() + DV_sum.std()/2).round(4)

        bins = [float('-inf'), lower, upper, float('inf')]
        DV_target = pd.cut(DV_sum, bins, labels = [0, 1, 2])
        df["target"] = DV_target
        
        # record target counts
        counts = df["target"].value_counts().sort_index()
        counts = counts.rename(f"model{i+1}")
        target_counts = pd.concat([target_counts, counts], axis = 1)

        # record target summary
        target_summary.append({
            "median": DV_sum.median().round(4),
            "std": DV_sum.std().round(4),
            "lower": lower, 
            "upper": upper
            })
    target_counts = target_counts.T
    target_summary = pd.DataFrame(target_summary, index = ["model1", "model2", "model3", "model4"])
    

    # rename columns
    for df in dfs:
        features = []
        for cols in df.columns:
            if cols[0] in ["sum", "target"]:
                features.append(cols[0])
            else:
                features.append(cols[2] + "_" + cols[3])
        df.columns = features


    # save data
    with pd.ExcelWriter(data_split, engine = "openpyxl", mode = "w") as writer: # 直接重寫一個新的檔案
        for i, df in enumerate(dfs):
            df.to_excel(writer, sheet_name = f"model{i+1}", index = False)

    
    return dfs