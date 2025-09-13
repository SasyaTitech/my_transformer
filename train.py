# import torch
# import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer, trainers, pre_tokenizers, normalizers, processors
from tokenizers.models import Unigram
from pathlib import Path

import json

with open("config.json", "r", encoding = "utf-8") as f:
    config = json.load(f)

def get_all_sentences(config, ds):
    fields = [f"{config["lang_tgt"]}", f"{config["lang_src"]}"]
    for row in ds:
        for lang in fields:
            #print(row)
            yield row['translation'][lang]
            

def get_or_build_tokenizer(config, ds):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(Unigram())
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFKC(), # 统一全角半角
            normalizers.Lowercase() # 英文全部视为小写
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement = "_",
            prepend_scheme = "always", # 在第一个词前面也补一个空格
        )
        trainer = trainers.UnigramTrainer(
            vocab_size = 50000,
            special_tokens = ["<pad>", "<unk>", "<s>", "</s>"],
            unk_token = "<unk>",
        )
        tokenizer.train_from_iterator(get_all_sentences(config, ds), trainer = trainer)
        # 把已经变成ids即将送去model的句子前后加上<s> </s>所对应的id
        # 这件事改成在Dataset（from pytorch）里干了
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_train = load_dataset("wmt17", f'{config["lang_tgt"]}-{config["lang_src"]}', split = 'train').shuffle(seed = 42).select(range(int(config["k_train"])))
    ds_validation = load_dataset("wmt17", f'{config["lang_tgt"]}-{config["lang_src"]}', split = 'validation')

    tokenizer = get_or_build_tokenizer(config, ds_train)

