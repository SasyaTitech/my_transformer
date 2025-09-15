from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # 别把tensor的大小写搞错了，tensor是函数式构造器（用普通函数创建并返回对象），Tensor是类构造器（用类名，调用其__init__来构造）
        self.sos_token = torch.tensor([tokenizer.token_to_id('<s>')], dtype = torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id('</s>')], dtype = torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('<pad>')], dtype = torch.int64)
        # 因为词表可能非常大，我开的是5w其实还好？，所以这里用int64
        # ？ 需要在dataset里把tensor移动到gpu

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer.encode(src_text).ids
        dec_input_tokens = self.tokenizer.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 
        # 在decoder环节输入我们只加sos不加eos，输出(label或者叫target)只加eos不加sos，所以少一个就ok

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')
        
        # Add sos and eos to the source text 然后再后面补pad补到上线
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )

        decoder_label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )   

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert decoder_label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input,
            "decoder_input" : decoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            # 这里还没有batch信息，三个维度分别对应着 1 - 注意力头数 2 - 序列长 3 - 序列长 对应每一个注意力头里每一个token能看到的token
            # 对于encoder里的每一个token都是统一的，别让他们看到padding就好，所以第二维度设成1来广播即可
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # (1, Seq_len) & (1, Seq_len, Seq_len) 可以被广播， 这里对于每一个token，mask是不一样的，每一个token在decoder只能看到自己前面的token
            # 这里因为要对每个token单独设置所以第二个维度存在
            # srds这里的pad是不是脱裤子放屁？pad不可能会在每个token的本体之前出现吧
            "decoder_label" : decoder_label,
            "src_text" : src_text,
            "tgt_text" : tgt_text
        } # 以字典形式回复之后会被自动加工成batch的形式


def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0
