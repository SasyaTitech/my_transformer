import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BilingualDataset, causal_mask
from model import build_transformer

from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer, trainers, pre_tokenizers, normalizers, processors
from tokenizers.models import Unigram
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm

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
        tokenizer.save(str(tokenizer_path))
        # 把已经变成ids即将送去model的句子前后加上<s> </s>所对应的id
        # 这件事改成在Dataset（from pytorch）里干了
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    ds_train = load_dataset("wmt17", f'{config["lang_tgt"]}-{config["lang_src"]}', split = 'train').shuffle(seed = 42).select(range(int(config["k_train"])))
    ds_val = load_dataset("wmt17", f'{config["lang_tgt"]}-{config["lang_src"]}', split = 'validation')

    tokenizer = get_or_build_tokenizer(config, ds_train)

    train_ds = BilingualDataset(ds_train, tokenizer, config["lang_src"], config["lang_tgt"], config["seq_len"])
    val_ds = BilingualDataset(ds_val, tokenizer, config["lang_src"], config["lang_tgt"], config["seq_len"])

    max_len_src = 0
    max_len_tgt = 0
    max_len_src_val = 0
    max_len_tgt_val = 0
    
    for item in ds_train:
        src_ids = tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    for item in ds_val:
        src_ids = tokenizer.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src_val = max(max_len_src_val, len(src_ids))
        max_len_tgt_val = max(max_len_tgt_val, len(tgt_ids))
    
    print(f'Max length of source sentences: {max(max_len_src, max_len_src_val)}')
    print(f'Max length of target sentences: {max(max_len_tgt, max_len_tgt_val)}')

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle = True)
    # 包装成dataloader

    return train_dataloader, val_dataloader, tokenizer
        

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model
    
def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents = True, exist_ok = True)

    train_dataloader, val_dataloader, tokenizer = get_ds(config) #batch size是写在config里的
    model = get_model(config, tokenizer.get_vocab_size(), tokenizer.get_vocab_size()).to(device) # 我们这里是中英一起分词的所以共用一个词表
    
    # Tensorboard可视化 这里还没弄明白
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    if config['preload']: # check 这里应该是优化到一半保存的样子？
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index = tokenizer.token_to_id('<pad>'), label_smoothing = 0.1).to(device)
    # label_smoothing不再把正确项设为1，而是会分给其他类一些概率，减少模型过度自信
    # 这里只是创建了交叉熵的对象

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f'Processing epoch {epoch:02d}')
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # run the tensor through the transformer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, Seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, Seq_len, tgt_vacab_size) 这里输出的是基于词表大小的logits

            label = batch['decoder_label'].to(device) #这种拿过来的tensor都要送去gpu再用 (B, seq_len) 这里的元素是ids

            # (B, seq_len, tgt_vocab_size) --> (B * Sqe_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer.get_vocab_size()), label.view(-1))
            # 交叉熵本来就被设计成： 输入 = 未归一化分数（logits）， 目标 = 类别索引
            # 这里相当于把批次x序列长度合成了一个大序列
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})


            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step': global_step
        }, model_filename) # 我们希望不仅恢复model的参数，也恢复优化器。？为什么

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)