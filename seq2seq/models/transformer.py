import math
import torch
import torch.nn as nn
from seq2seq import utils
from seq2seq.models import register_model, register_model_architecture
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
import sentencepiece as spm


class RotaryPositionEncoding(nn.Module):
    """RoPE: Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos 和 sin
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """预计算cos和sin缓存"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, :, :], persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int = None):
        """返回 cos 和 sin 用于旋转"""
        if seq_len is None:
            seq_len = x.shape[1]
        
        # 如果序列长度超过缓存,重新构建
        if seq_len > self.cos_cached.shape[1]:
            self._build_cache(seq_len)
        
        return (
            self.cos_cached[:, :seq_len, :],
            self.sin_cached[:, :seq_len, :]
        )


def rotate_half(x):
    """旋转输入张量的一半维度"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """应用旋转位置编码到 query 和 key"""
    # q, k shape: (batch, heads, seq_len, head_dim)
    # cos, sin shape: (1, seq_len, dim)
    
    # 调整 cos 和 sin 的形状以匹配 q 和 k
    cos = cos.unsqueeze(1)  # (1, 1, seq_len, dim)
    sin = sin.unsqueeze(1)  # (1, 1, seq_len, dim)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@register_model('transformer')
class TransformerModel(Seq2SeqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """ Add model-specific arguments to the parser. """
        parser.add_argument('--encoder-embed-path', type=str, help='Path to pre-trained encoder embeddings')
        parser.add_argument('--decoder-embed-path', type=str, help='Path to pre-trained decoder embeddings')
        parser.add_argument('--encoder-dropout', type=float, default=0.0, help='dropout probability for encoder layers')
        parser.add_argument('--decoder-dropout',type=float, default=0.0,help='dropout probability for decoder layers')
        
        parser.add_argument('--dim-embedding', type=int, default=512, help='embedding dimension for both encoder and decoder')
        parser.add_argument('--attention-heads', type=int, default=8, help='number of attention heads')
        parser.add_argument('--dim-feedforward-encoder', type=int, default=2048, help='dimension of feed-forward layers for encoder')
        parser.add_argument('--dim-feedforward-decoder', type=int, default=2048, help='dimension of feed-forward layers for decoder')
        parser.add_argument('--max-seq-len', type=int, default=128, help='maximum sequence length')
        parser.add_argument('--n-encoder-layers', type=int, default=6, help='number of encoder layers')
        parser.add_argument('--n-decoder-layers', type=int, default=6, help='number of decoder layers')
        parser.add_argument('--rope-base', type=int, default=10000, help='RoPE base frequency')
        
    @classmethod
    def build_model(cls, args, src_tokenizer, tgt_tokenizer):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_tokenizer)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_tokenizer)

        encoder = TransformerEncoder(
            src_tokenizer=src_tokenizer,
            dim_embed=args.dim_embedding,
            dropout=args.encoder_dropout,
            max_seq_len=args.max_seq_len,
            n_attention_heads=args.attention_heads,
            dim_ff=args.dim_feedforward_encoder,
            pretrained_embedding=encoder_pretrained_embedding,
            n_encoder_layers=args.n_encoder_layers,
            rope_base=args.rope_base,
        )
        decoder = TransformerDecoder(
            tgt_tokenizer=tgt_tokenizer,
            dim_embed=args.dim_embedding,
            n_attention_heads=args.attention_heads,
            dropout=args.decoder_dropout,
            max_seq_len=args.max_seq_len,
            n_decoder_layers=args.n_decoder_layers,
            dim_ff=args.dim_feedforward_decoder,
            pretrained_embedding=decoder_pretrained_embedding,
            use_cuda=args.cuda,
            rope_base=args.rope_base,
        )
        return cls(encoder, decoder)

    def forward(self, src, src_mask, trg, trg_pad_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, trg, trg_pad_mask)


class TransformerEncoder(Seq2SeqEncoder):
    '''Encoder = token embedding + RoPE -> a stack of N EncoderBlock -> layer norm'''
    def __init__(self,
                 src_tokenizer: spm.SentencePieceProcessor,
                 dim_embed,
                 dropout,
                 max_seq_len,
                 n_attention_heads,
                 dim_ff,
                 pretrained_embedding,
                 n_encoder_layers,
                 rope_base=10000):
        super().__init__(src_tokenizer)

        self.src_vocab_size = src_tokenizer.GetPieceSize()
        self.dim_embed = dim_embed
        
        # Token embedding (不再需要位置embedding)
        self.tok_embed = nn.Embedding(self.src_vocab_size, dim_embed)
        
        # RoPE 位置编码
        head_dim = dim_embed // n_attention_heads
        self.rope = RotaryPositionEncoding(head_dim, max_seq_len, rope_base)
        
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(dim_embed, dropout, n_attention_heads, dim_ff, self.rope) 
            for _ in range(n_encoder_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(dim_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x = self.dropout(x)
        
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        
        return self.norm(x)


class EncoderBlock(nn.Module):
    '''EncoderBlock: self-attention with RoPE -> feed-forward layer'''
    def __init__(self, dim_embed, dropout, n_heads, dim_ff, rope):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttentionRoPE(n_heads, dim_embed, dropout, rope)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embed, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_embed)
        )
        self.residual1 = ResidualConnection(dim_embed, dropout)
        self.residual2 = ResidualConnection(dim_embed, dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        return self.residual2(x, self.feed_forward)


class TransformerDecoder(Seq2SeqDecoder):
    '''Decoder = token embedding + RoPE -> a stack of N DecoderBlock -> output layer'''
    def __init__(self,
                 tgt_tokenizer: spm.SentencePieceProcessor,
                 dim_embed: int,
                 n_attention_heads: int,
                 dropout: float,
                 max_seq_len: int,
                 n_decoder_layers: int,
                 dim_ff: int,
                 pretrained_embedding,
                 use_cuda: bool,
                 rope_base=10000):
        super().__init__(tgt_tokenizer)
        self.tgt_vocab_size = tgt_tokenizer.GetPieceSize()
        self.dim_embed = dim_embed
        
        # Token embedding (不再需要位置embedding)
        self.tok_embed = nn.Embedding(self.tgt_vocab_size, dim_embed)
        
        # RoPE 位置编码
        head_dim = dim_embed // n_attention_heads
        self.rope = RotaryPositionEncoding(head_dim, max_seq_len, rope_base)
        
        self.dropout = nn.Dropout(dropout)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(dim_embed, n_attention_heads, dropout, dim_ff, self.rope) 
            for _ in range(n_decoder_layers)
        ])
        self.norm = nn.RMSNorm(dim_embed)
        self.linear = nn.Linear(dim_embed, self.tgt_vocab_size)
        self.device = torch.device("cuda" if use_cuda else "cpu")
    
    def future_mask(self, seq_len: int):
        '''mask out tokens at future positions'''
        mask = (torch.triu(torch.ones(seq_len, seq_len, requires_grad=False), diagonal=1)!=0).to(self.device)
        return mask.view(1, 1, seq_len, seq_len)

    def forward(self, encoder_out: torch.Tensor, src_mask: torch.Tensor, trg: torch.Tensor, trg_pad_mask: torch.Tensor):
        seq_len = trg.size(1)
        trg_mask = torch.logical_or(trg_pad_mask, self.future_mask(seq_len))
        
        x = self.tok_embed(trg)
        x = self.dropout(x)
        
        for layer in self.decoder_blocks:
            x = layer(encoder_out, src_mask, x, trg_mask)
        
        x = self.norm(x)
        logits = self.linear(x)
        return logits


class MultiHeadedAttentionRoPE(nn.Module):
    """Multi-head attention with RoPE position encoding"""
    def __init__(self, n_heads: int, dim_embed: int, dropout: float, rope: RotaryPositionEncoding):
        super().__init__()
        assert dim_embed % n_heads == 0
        
        self.d_k = dim_embed // n_heads
        self.dim_embed = dim_embed
        self.h = n_heads
        self.rope = rope
        
        self.WQ = nn.Linear(dim_embed, dim_embed)
        self.WK = nn.Linear(dim_embed, dim_embed)
        self.WV = nn.Linear(dim_embed, dim_embed) 
        self.linear = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0)
        seq_len_q = x_query.size(1)
        seq_len_k = x_key.size(1)
        
        # 1) Linear projections
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        key = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
        
        # 2) Apply RoPE to query and key
        cos_q, sin_q = self.rope(x_query, seq_len_q)
        cos_k, sin_k = self.rope(x_key, seq_len_k)
        query, key = apply_rotary_pos_emb(query, key, cos_q, sin_q)
        
        # 3) Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 4) Mask
        if mask is not None:
            mask = mask.unsqueeze(1) if mask.dim() == 3 else mask
            scores = scores.masked_fill(mask, float('-inf'))
        
        p_atten = torch.nn.functional.softmax(scores, dim=-1)
        p_atten = self.dropout(p_atten)
        
        # 5) Apply attention to values
        x = torch.matmul(p_atten, value)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.dim_embed)
        
        return self.linear(x)


class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.RMSNorm(dim)

    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))


class DecoderBlock(nn.Module):
    '''DecoderBlock with RoPE: self-attention -> cross-attention -> feed-forward'''
    def __init__(self, dim_embed, n_heads, dropout, dim_ff, rope):
        super().__init__()
        self.atten1 = MultiHeadedAttentionRoPE(n_heads, dim_embed, dropout, rope)
        self.atten2 = MultiHeadedAttentionRoPE(n_heads, dim_embed, dropout, rope)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_embed, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_embed)
        )
        self.residuals = nn.ModuleList([
            ResidualConnection(dim_embed, dropout) for _ in range(3)
        ])

    def forward(self, memory, src_mask, decoder_layer_input, trg_mask):
        x = memory
        y = decoder_layer_input
        
        # Self-attention with causal mask
        y = self.residuals[0](y, lambda y: self.atten1(y, y, y, mask=trg_mask))
        # Cross-attention
        y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
        # Feed-forward
        return self.residuals[2](y, self.feed_forward)


@register_model_architecture('transformer', 'transformer')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.encoder_dropout = getattr(args, 'encoder_dropout', 0.0)
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.0)

    args.dim_embedding = getattr(args, 'dim_embedding', 512)
    args.attention_heads = getattr(args, 'attention_heads', 8)
    args.dim_feedforward_encoder = getattr(args, 'dim_feedforward_encoder', 2048)
    args.dim_feedforward_decoder = getattr(args, 'dim_feedforward_decoder', 2048)
    args.max_seq_len = getattr(args, 'max_seq_len', 512)
    args.n_encoder_layers = getattr(args, 'n_encoder_layers', 6)
    args.n_decoder_layers = getattr(args, 'n_decoder_layers', 6)
    args.rope_base = getattr(args, 'rope_base', 10000)
