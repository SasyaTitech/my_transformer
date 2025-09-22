from pathlib import Path

def get_config():
    return {
        "lang_src" : "en",
        "lang_tgt" : "zh",
        "tokenizer_file" : "./unigram.json",
        "k_train" : "10000",
        "seq_len" : 500,
        "batch_size" : 8,
        "d_model" : 512,
        "lr": 10**-4,
        "model_folder" : "weights",
        "model_basename" : "tmodel_",
        "preload" : None,
        "num_epochs" : 20,
        "experiment_name" : "runs/tmodel" # 储存loss之类的实验数据
    }

def get_weights_file_path(config, _epoch:str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{_epoch}.pt"
    return str(Path('.') / model_folder / model_filename)