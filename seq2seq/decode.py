import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, 
           max_out_len: int, tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """
    Decodes a sequence without teacher forcing using greedy decoding.
    
    Args:
        model: The Seq2Seq model (encoder + decoder)
        src_tokens: Source tokens (batch_size, src_len)
        src_pad_mask: Source padding mask
        max_out_len: Maximum output length
        tgt_tokenizer: Target tokenizer
        args: Training arguments
        device: Device (cuda/cpu)
    
    Returns:
        predicted_tokens: List of token IDs for each sequence in batch
    """
    model.eval()
    
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    
    # ============================================================
    # 🔧 FIX 1: 初始化生成序列为 [BOS]
    # ============================================================
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # ============================================================
    # 🔧 FIX 2: 获取最大序列长度限制
    # ============================================================
    if hasattr(model.decoder, 'max_seq_len'):
        max_len = model.decoder.max_seq_len
    elif hasattr(model.decoder, 'rope') and hasattr(model.decoder.rope, 'max_seq_len'):
        max_len = model.decoder.rope.max_seq_len
    else:
        max_len = 512  # 默认值
    
    # ============================================================
    # 🔧 FIX 3: 编码源序列 (只需要做一次)
    # ============================================================
    with torch.no_grad():
        encoder_out = model.encoder(src_tokens, src_pad_mask)
    
    # ============================================================
    # 自回归生成循环
    # ============================================================
    for t in range(max_out_len):
        # 检查是否超过最大长度
        current_len = generated.size(1)
        if current_len >= max_len:
            break
        
        # ============================================================
        # 🔧 FIX 4: 正确构造 padding mask
        # Shape: (batch_size, 1, 1, current_len) for broadcasting
        # ============================================================
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(1)
        
        # ============================================================
        # 🔧 FIX 5: 调用 decoder (不是整个 model)
        # 避免重复编码 encoder
        # ============================================================
        with torch.no_grad():
            # 直接调用 decoder,传入已编码的 encoder_out
            logits = model.decoder(encoder_out, src_pad_mask, generated, trg_pad_mask)
        
        # ============================================================
        # 🔧 FIX 6: 获取最后一个时间步的 logits
        # Shape: (batch_size, vocab_size)
        # ============================================================
        next_token_logits = logits[:, -1, :]
        
        # ============================================================
        # 🔧 FIX 7: Greedy decoding - 选择概率最高的 token
        # ============================================================
        next_tokens = next_token_logits.argmax(dim=-1)  # (batch_size,)
        
        # ============================================================
        # 🔧 FIX 8: 只为未完成的序列添加新 token
        # 已完成的序列添加 PAD
        # ============================================================
        next_tokens = torch.where(finished, PAD, next_tokens)
        
        # 拼接到生成序列
        generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
        
        # ============================================================
        # 🔧 FIX 9: 更新完成状态
        # ============================================================
        finished = finished | (next_tokens == EOS)
        
        # 如果所有序列都完成,提前退出
        if finished.all():
            break
    
    # ============================================================
    # 🔧 FIX 10: 后处理 - 移除 BOS,截断 EOS 后的内容
    # ============================================================
    predicted_tokens = []
    for seq in generated.tolist():
        # 跳过第一个 BOS token
        seq = seq[1:]
        
        # 如果有 EOS,截断 EOS 及之后的内容
        if EOS in seq:
            eos_idx = seq.index(EOS)
            seq = seq[:eos_idx]  # 不包含 EOS
        
        # 移除所有 PAD tokens
        seq = [tok for tok in seq if tok != PAD]
        
        predicted_tokens.append(seq)
    
    return predicted_tokens
