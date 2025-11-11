import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        # Create target padding mask with correct batch dimension
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        # Ensure trg_pad_mask has shape (batch_size, seq_len)
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        # Forward pass: use only the generated tokens so far
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens


#Beam Search Decoding
def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5):
    """Beam Search decoding compatible with Transformer (no hidden state)."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]  # (sequence, cumulative log prob)

    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            # stop expanding finished sequences
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            # forward pass
            with torch.no_grad():
                trg_pad_mask = (seq == PAD).unsqueeze(1).unsqueeze(2)
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        # keep only the top-k beams overall
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # early stopping if all beams end with EOS
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break

    best_seq, _ = beams[0]
    return best_seq.squeeze(0).tolist()
