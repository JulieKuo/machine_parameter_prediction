import os, json, sys, pickle
import pandas as pd
from statistics import mode
from traceback import format_exc
from log_config import Log
from tools import *



class Predict():
    def __init__(self, root, input_, logging):
        self.logging  = logging
        self.model_id = input_["model_id"]
        self.upload_time = input_["upload_time"]
        self.machine_id = input_["machine_id"]


        # 取得input
        input_json = os.path.join(input_["path"], "input.json")
        self.output_json = os.path.join(input_["path"], "output.json")
        with open(input_json, encoding = "utf-8") as f:
            self.input_ = json.load(f)


        # 取得model位置
        self.ng4_5_detail = os.path.join(root, "data", "train", self.model_id[0], "ng4_5", "model")
        self.ng1_3_detail = os.path.join(root, "data", "train", self.model_id[1], "ng1_3", "model")


        # NG歷史紀錄
        pred_path = os.path.join(root, "data", "predict")        
        os.makedirs(pred_path, exist_ok = True)
        self.ng1_3_history_path = os.path.join(pred_path, "NG1_3.json")

        if os.path.exists(self.ng1_3_history_path): # 如果檔案存在，則讀取檔案內容    
            with open(self.ng1_3_history_path, encoding="utf-8") as f:
                self.ng1_3_history = json.load(f)
        else: # 如果檔案不存在，則創建一個新的dict
            self.ng1_3_history = {}


        # 取得config
        config_path = os.path.join(root, "prog", "config.json")
        with open(config_path, encoding = "utf-8") as f:
            config = json.load(f)
        self.model_cols = config["model_cols"] # set columns for each models
        self.directions = config["directions"]
    
    
    
    def run(self):
        try:            
            self.logging.info("Check NG or OK.")
            df = pd.DataFrame(self.input_, index = [0])
            counts = len(set([i.split("_")[0] for i in self.input_.keys() if i != "time"])) # 共多少點
            
            target_ER = [model_col["primary"][0] + "_ER" for model_col in self.model_cols.values()]
            target_value = df.loc[0, target_ER]
            ng_col = target_value[target_value != 0].index.str.replace('_ER', '').tolist()

            ng_flag = (target_value != 0).any()
            if not ng_flag: # OK
                result = {    
                    "status":   "success",
                    "upload_time": self.upload_time,
                    "model_id": self.model_id,
                    "result": 0,
                    "counts": counts
                    }
            
            else: # NG
                self.logging.info("Choose the adjustment method.")
                # calculate total ER by each side
                left_ER_col = [col for col in df.columns if ("ER" in col) and ("左邊" in col) and ("右邊" not in col)]
                right_ER_col = [col for col in df.columns if ("ER" in col) and ("右邊" in col) and ("左邊" not in col)]
                left  = df[left_ER_col].sum(axis = 1)
                right = df[right_ER_col].sum(axis = 1)

                # choose the adjustment method
                error = abs(right - left)[0]
                ng4_5_flag = (error <= 0.001)
                
                
                # predict
                pred = {}
                if ng4_5_flag:
                    self.logging.info("NG4_5 model.")
                    model = pickle.load(open(os.path.join(self.ng4_5_detail, "model.pkl"), "rb"))
                    features = pickle.load(open(os.path.join(self.ng4_5_detail, "features.pkl"), "rb"))
                    X_test = df[features]
                    test_pred = model.predict(X_test)[0]
                    pred.update({
                        "model": "ng4_5",
                        "adjust": test_pred
                    })

                else:
                    self.logging.info("NG1_3 model.")
                    pred.update({
                        "model": "ng1_3",
                        "adjust": {},
                    })
                    for i, model_col in enumerate(self.model_cols.values(), start = 1):
                        primary = model_col["primary"][0]
                        if primary in ng_col:
                            model = pickle.load(open(os.path.join(self.ng1_3_detail, f"model{i}.pkl"), "rb"))        
                            scaler = pickle.load(open(os.path.join(self.ng1_3_detail, f"scaler{i}.pkl"), "rb"))
                            features = pickle.load(open(os.path.join(self.ng1_3_detail, f"features{i}.pkl"), "rb"))

                            X_test = df[features]
                            X_test = scaler.transform(X_test)
                            test_pred = model.predict(X_test).argmax(axis = 1)[0]
                            pred["adjust"][primary] = int(test_pred + 1)  # NG1: -0.001, NG2: 0, NG3: +0.001
                        else:            
                            pred["adjust"][primary] = 2 # 沒NG補2


                # save NG hisitory
                if pred["model"] == "ng1_3":
                    if (self.machine_id in self.ng1_3_history):
                        self.ng1_3_history[self.machine_id][self.upload_time] = pred["adjust"]
                    else:
                        self.ng1_3_history[self.machine_id] = {self.upload_time: pred["adjust"]}

                    with open(self.ng1_3_history_path, 'w', encoding = 'utf-8') as file:
                        json.dump(self.ng1_3_history, file, indent = 4, ensure_ascii = False)


                self.logging.info(f"Build result.")
                # bulid compensate
                mode_pred = []
                if pred["model"] == "ng1_3":
                    compensate = {}
                    for primary, adjust in pred["adjust"].items():
                        if adjust == 2: # NG1: -0.001, NG2: 0, NG3: +0.001，2不須補償
                            continue
                        
                        point = primary.split("距離")[0] # point

                        # direction and no
                        if "A0" in primary:
                            direction = self.directions["A0"]["name"]
                            no = self.directions["A0"]["no"]
                        elif "A180" in primary:
                            direction = self.directions["A180"]["name"]
                            no = self.directions["A180"]["no"]
                        else:
                            direction = self.directions["A0+A180"]["name"]
                            no = self.directions["A0+A180"]["no"]

                        # side
                        if ("右邊" in primary) and ("左邊" in primary):
                            side = "left_right"
                        elif "左邊" in primary:
                            side = "left"
                        elif "右邊" in primary:
                            side = "right"
                        else:
                            side = "left_right"

                        adjust1 = -0.001 if (adjust == 1) else 0.001 # NG1: -0.001, NG2: 0, NG3: +0.001

                        if (direction in compensate) and (side in compensate[direction]):
                            compensate[direction][side] = {"no": no, point: adjust1}
                        else:
                            compensate[direction] = {
                                side:{
                                    "no": no,
                                    point: adjust1
                                }
                            }
                        
                        mode_pred.append(adjust) # 最後result從所有adjust中不為NG2的取眾數
                        
                    result = 2 if mode_pred == [] else mode(mode_pred)
                else:
                    result, compensate = (4, "NC_FIX_1") if pred["adjust"] == 0 else (5, "NC_FIX_2")  


                result = {    
                    "status":   "success",
                    "upload_time": self.upload_time,
                    "model_id": self.model_id,
                    "result": result,
                    "counts": counts
                    }
                
                if compensate:
                    result["compensate"] = compensate
            
        
        except:
            message = format_exc()
            result  = error(self.logging, message, self.model_id)


        finally:
            self.logging.info(f'Save output to {self.output_json}')
            with open(self.output_json, 'w', encoding = 'utf-8') as file:
                json.dump(result, file, indent = 4, ensure_ascii = False)



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "predict.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "predict")
    
    logging.info("-"*200)
    # logging.info(f"root: {root}")
    

    input_ = get_input(sys.argv, logging)
    logging.info(f"input = {input_}")


    predict = Predict(root, input_, logging)
    predict.run()
            
    log.shutdown()