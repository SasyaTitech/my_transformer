import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model : int, vocab_size : int) -> None:
        # word embedding 告知其词向量维度以及总词汇量
        # 这里我们预计是做512维的词向量

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        # nn提供的这个embedding，起到了一个把每个序号固定映射到一个d_model维度的向量的一个字典功能
        # 具体的映射是甚么，得由模型来学习

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout) # droupout直接用现成的nn层了
        ## droupout 是保持期望同时屏蔽一定比例的进入的维度（为0）达到防止过拟合（正则化）的效果

        # 制作位置编码矩阵
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #（ (0), (1), (2), ... , (seq_len - 1) )  -> (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).unsqueeze(0) #把指数用对数形式稳定实现，更加数值稳定和高效
        # 这里是一个0到d_model，步长为2的向量，长度为d_model / 2 向上取整 (0, 2, 4, 6, ... , d_model - 2(或-1) ) -> (ciel(d_model / 2) , )

        # apply sin to even positions, cos to the odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) #其实这里如果d_model是奇数会遭重，但是d_model内定为512了所以没问题

        pe = pe.unsqueeze(0) #加上batch的维度

        self.register_buffer('pe', pe) 
        # 1. 可保存和读取 2.自动迁移设备 3.

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], ]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10 ** -6) -> None:
        super().__init__()
        self.eps = eps # 防止分母为0
        self.alpha = nn.Parameter(torch.ones(1)) # 乘
        self.bias = nn.Parameter(torch.zeros(1)) # 加 
        # 这里alpha和bias被注册为了nn.Parameter所以会被学习和更新

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
        # 通过学习恢复每个通道的幅度和偏置

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # for W1 and B1
        self.dropout = nn.Dropout(dropout)
        nn.linear_2 = nn.Linear(d_ff, d_model) # for W2 and B2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h : int, dropout : float) -> None: 
        super().__init__()
        self.d_model = d_model
        self.h = h # d_model要被分成h份所以我们要确定这玩意能整除
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # 省的自己去初始化以及声明梯度追踪了...
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    #弄个函数方便处理矩阵乘法
    @staticmethod #不需要实例也能调取方法
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # @ 就是乘最后两维罢了
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # 这个缩放很重要，确保点乘的方差为1，避免softmax梯度消失
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # mask住的设很小的值就好了
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores) # 随机不看一些位置

        return (attention_scores @ value), attention_scores # 后者是用来可视化的，看看哪个词注意了哪个词，这玩意对解释性来说很重要



    def forward(self, q, k, v, mask): #为啥这里有qkv
        query = self.w_q(q) # (Batch, Seq_len, d_model) -> (Batch, d_model, d_model) 这里其实已经分出不同的头了
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # 把最后的d_model个要素划分成 h 个 d_k 维向量
        # (Batch, Seq_len, d_model) -> (Batch, Seq_len, h, d_k) -> (Batch, h, Seq_len, d_k) 每个头的query都视为浓缩了整个seq的信息
        key = key.view(key.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # -1 代表自动填充
        # contiguous让x在内存上先连续排列，大概是什么pytorch的计算细节可以先忽略
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        # 实际上这里运行两条线路
        # 1. 先过norm，再过self attention， 再dropout  <- 这里的顺序和论文是反过来的 
        # 2. x的本体，也就是residual残差

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # 两个一组的残差

    def forward(self, x, src_mask): # 这个massk主要是为了屏蔽padding
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask)) # 对于encoder来说，qkv都按x本身就行
        x = self.residual_connections[1](x, lambda x : self.feed_forward_block(x))
        return x

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # 之前设计的层都是先norm后操作，这次在最后再加上一层norm
        

# 事实上对于当今的gpt等decoder only的模型，能更平等的对待原本的输入和作为输入的之前的输出

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block : MultiHeadAttentionBlock, cross_attention_block : MultiHeadAttentionBlock, feed_forward_block : FeedForwardBlock, dropout : float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
         x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
         x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
         x = self.residual_connections[2](x, self.feed_forward_block)
         return x # 这个cross attention导致参考部分和已经输出的部分实际上是不平等的地位。
         # 因为这是一个翻译项目所以并没有什么问题

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        # 把终于探出头的词向量还原到词表的序号上

    def forward(self, x):
        # (batch, Seq_len, d_model) -> (batch, Seq_len, Vocab_Size) 同时每个位置的值对应其可能性的大小
        return torch.log_softmax(self.proj(x), dim = -1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    # 分离写一下encode和decode，主要是为了复用encode的结果以及方便之后可视化
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)


# 目前为止，我们的模型套娃，默认输入的都是已经实例化的层，但是我们目前还没有实例化
# 这里的src_seq_len是固定的..?应该是固定的吗
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff : int = 2048) -> Transformer:
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    # 虽然其实没必要弄两个独立的pos_encode但是作为教学这样更清楚，也方便之后可视化

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout) # 这里的d_ff是ffw的中间层维度
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N): # 可以看到模型对象本体创建的时候是模型基本属性相关的超参数，而模型对其他的东西如何操作是卸载fw里的
        # 但是此处的设计里residual layer是被隐藏的，实例是在encoder/decoder的实例化中被创建的
        # 而residual layer的创建实际上也并没有直接传参实例化的attention block之类，而是设计成作为fw的参数传入
        # 这么设计是否有点混乱？为什么要这么设计？
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # 把所有维度超过1的超参数以xavier_uniform方法初始化（
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
    