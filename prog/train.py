import os, json, sys, warnings
import pandas as pd
from traceback import format_exc
from split_data import split_data
from train_ng1_3 import ng1_3
from train_ng4_5 import ng4_5
from log_config import Log
from tool import *
from datetime import datetime

warnings.filterwarnings("ignore")



class Model():
    def __init__(self, root, input_, logging):
        self.logging  = logging
        self.model_id = input_["model_id"]
        self.start_time = datetime.strptime(input_["start_time"] + " 00:00:00", '%Y-%m-%d %H:%M:%S')
        self.end_time = datetime.strptime(input_["end_time"] + " 23:59:59", '%Y-%m-%d %H:%M:%S')
        self.model_type = input_["model_type"]


        # 取得train位置
        train_path = os.path.join(root, "data", "train")        
        os.makedirs(train_path, exist_ok = True)
        self.data = os.path.join(train_path, "train.xlsx")        
        self.data_split = os.path.join(train_path, "train_split.xlsx")
        self.output_json = os.path.join(train_path, "output.json")


        # 取得current_model位置
        self.ng4_5_model_path = os.path.join(root, "data", "train", self.model_id, "ng4_5")
        os.makedirs(self.ng4_5_model_path, exist_ok = True)
        
        self.ng4_5_model_detail = os.path.join(self.ng4_5_model_path, "model")
        os.makedirs(self.ng4_5_model_detail, exist_ok = True)


        # 取得next_model位置
        self.ng1_3_model_path = os.path.join(root, "data", "train", self.model_id, "ng1_3")
        os.makedirs(self.ng1_3_model_path, exist_ok = True)
        
        self.ng1_3_model_detail = os.path.join(self.ng1_3_model_path, "model")
        os.makedirs(self.ng1_3_model_detail, exist_ok = True)


        # 取得config
        config_path = os.path.join(root, "prog", "config.json")
        with open(config_path, encoding = "utf-8") as f:
            config = json.load(f)
        self.model_cols = config["model_cols"] # set columns for each models
    
    
    
    def run(self):
        try:
            self.logging.info(f"Get data from {self.data}")
            df_raw = pd.read_excel(self.data, sheet_name = "simulation", header = [0, 1, 2, 3])
            df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0])
            self.df_raw = df_raw[(self.start_time <= df_raw.iloc[:, 0]) & (df_raw.iloc[:, 0] <= self.end_time)]
            print(f"raw:      columns: {self.df_raw.shape[1]}, length: {self.df_raw.shape[0]}")
            if self.df_raw.empty:
                raise NoDataFoundException
            
            
            self.logging.info(f"Split data to train the next adjusted model.")
            dfs = split_data(self.df_raw, self.data_split, self.model_cols)

            
            self.logging.info(f"Train the next adjusted model.")
            ng1_3_score = ng1_3(dfs, self.ng1_3_model_path, self.ng1_3_model_detail)
            

            self.logging.info(f"Train the current adjusted model.")
            ng4_5_score = ng4_5(self.df_raw, self.ng4_5_model_path, self.ng4_5_model_detail)

            use_index = int(ng1_3_score["test"].nlargest(3).index[-1])
            ng1_3_score1 = ng1_3_score.loc[use_index, "test"]
            ng4_5_score1 = ng4_5_score.loc[0, "test"]
            accuracy = ng1_3_score1 if self.model_type == 0 else ng4_5_score1

            result = {
                "status":     "success",
                "model_id":   self.model_id,
                "model_type": self.model_type,
                "accuracy": accuracy,
                "use_index": use_index + 1,
                "ng1_3_score": ng1_3_score1,
                "ng4_5_score": ng4_5_score1
                }


        except (pd.errors.EmptyDataError, NoDataFoundException):
            message = "No data is available."
            result  = error(self.logging, message, self.model_id)
        

        except FileNotFoundError:
            message = "File not found."
            result  = error(self.logging, message, self.model_id)
        
        
        except:
            message = format_exc()
            result  = error(self.logging, message, self.model_id)


        finally:
            self.logging.info(f'Save output to {self.output_json}')
            with open(self.output_json, 'w', encoding = 'utf-8') as file:
                json.dump(result, file, indent = 4, ensure_ascii = False)
    


class NoDataFoundException(Exception):
    pass



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "train.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "train")
    
    logging.info("-"*200)
    # logging.info(f"root: {root}")
    

    input_ = get_input(sys.argv, logging)
    logging.info(f"input = {input_}")


    model = Model(root, input_, logging)
    model.run()
            
    log.shutdown()