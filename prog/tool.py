import base64, json
from traceback import format_exc



def get_input(argv, logging):
    try:   
        if len(argv) == 2:
            input_ = argv[1]
            input_ = base64.b64decode(input_).decode('utf-8')
            input_ = json.loads(input_)

            return input_
        else:
            logging.info("Input parameter error.")
    except:
        logging.error(format_exc())



def error(logging, message, model_id):
    logging.error(message)
    result = {
        "status": "fail",
        "model_id": model_id,
        "reason": message
        }
    
    return result