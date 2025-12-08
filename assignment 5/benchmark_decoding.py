import os
import time
import argparse
import torch
import sentencepiece as spm
from torch.serialization import default_restore_location
from seq2seq import models, utils
from seq2seq.decode import decode as fast_decode, beam_search_decode as fast_beam_search_decode

# --- Define Slow Versions (Original Code) ---

def slow_decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, args, device):
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    for t in range(max_out_len):
        max_len = model.decoder.pos_embed.size(1)
        if generated.size(1) > max_len:
            generated = generated[:, :max_len]
        trg_pad_mask = (generated == PAD).unsqueeze(1).unsqueeze(2)
        # Inefficient: Re-running the whole model (encoder + decoder)
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask).to(device)
        next_token_logits = output[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tokens], dim=1)
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens

def slow_beam_search_decode(model, src_tokens, src_pad_mask, max_out_len, tgt_tokenizer, args, device, beam_size=5, alpha=0.7):
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue
            with torch.no_grad():
                max_len = model.decoder.pos_embed.size(1)
                if seq.size(1) > max_len:
                    seq = seq[:, :max_len]
                trg_pad_mask = (seq == PAD)[:, None, None, :]
                # Inefficient: Re-running the whole model
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                new_score = score + topk_log_probs[:, k].item()
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break
    best_seq, _ = beams[0]
    return [best_seq.squeeze(0).tolist()]

# --- Benchmark Setup ---

def run_benchmark():
    # Hardcoded arguments for reproduction
    args = argparse.Namespace(
        cuda=True,
        seed=42,
        input='cz-en/data/raw/test.cz',
        src_tokenizer='cz-en/tokenizers/cz-bpe-8000.model',
        tgt_tokenizer='cz-en/tokenizers/en-bpe-8000.model',
        checkpoint_path='cz-en/checkpoints/checkpoint_best.pt',
        batch_size=1,
        max_len=300,
        encoder_dropout=0.1,
        decoder_dropout=0.1,
        dim_embedding=256,
        attention_heads=4,
        dim_feedforward_encoder=1024,
        dim_feedforward_decoder=1024,
        max_seq_len=300,
        n_encoder_layers=3,
        n_decoder_layers=3,
        encoder_embed_path=None,
        decoder_embed_path=None
    )

    # Load Model
    print("Loading model...")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        map_loc = None
    else:
        map_loc = 'cpu'
        args.cuda = False
    
    state_dict = torch.load(args.checkpoint_path, map_location=map_loc, weights_only=False)
    # Merge args
    saved_args = state_dict['args']
    for k, v in vars(saved_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)
            
    utils.init_logging(args)
    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)
    
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    if args.cuda:
        model = model.cuda()
    model.eval()
    model.load_state_dict(state_dict['model'])
    
    # Prepare Data (First 100 sentences)
    print("Preparing data...")
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()][:100]
    
    src_encoded = [torch.tensor(src_tokenizer.Encode(line, out_type=int, add_eos=True)) for line in src_lines]
    max_seq_len = min(model.encoder.pos_embed.size(1), args.max_len)
    src_encoded = [s if len(s)<=max_seq_len else s[:max_seq_len] for s in src_encoded]
    
    device = torch.device('cuda' if args.cuda else 'cpu')
    PAD = src_tokenizer.pad_id()

    def get_batch(idx):
        src_ids = src_encoded[idx].unsqueeze(0).to(device)
        src_pad_mask = (src_ids == PAD).unsqueeze(1).unsqueeze(2)
        return src_ids, src_pad_mask

    log_file = open("benchmark_results.log", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Running benchmark on {len(src_encoded)} sentences...")

    # --- Test Greedy Decoding ---
    log("\n--- Greedy Decoding ---")
    
    # Slow
    start_time = time.time()
    for i in range(len(src_encoded)):
        src_ids, src_pad_mask = get_batch(i)
        slow_decode(model, src_ids, src_pad_mask, args.max_len, tgt_tokenizer, args, device)
    slow_time = time.time() - start_time
    log(f"Original (Slow) Greedy: {slow_time:.2f}s")

    # Fast
    start_time = time.time()
    for i in range(len(src_encoded)):
        src_ids, src_pad_mask = get_batch(i)
        fast_decode(model, src_ids, src_pad_mask, args.max_len, tgt_tokenizer, args, device)
    fast_time = time.time() - start_time
    log(f"Optimized (Fast) Greedy: {fast_time:.2f}s")
    log(f"Speedup: {slow_time / fast_time:.2f}x")

    # --- Test Beam Search ---
    beam_sizes = [1, 3, 5]
    for k in beam_sizes:
        log(f"\n--- Beam Search (k={k}) ---")
        
        # Slow
        start_time = time.time()
        for i in range(len(src_encoded)):
            src_ids, src_pad_mask = get_batch(i)
            slow_beam_search_decode(model, src_ids, src_pad_mask, args.max_len, tgt_tokenizer, args, device, beam_size=k)
        slow_time = time.time() - start_time
        log(f"Original (Slow) Beam Search: {slow_time:.2f}s")

        # Fast
        start_time = time.time()
        for i in range(len(src_encoded)):
            src_ids, src_pad_mask = get_batch(i)
            fast_beam_search_decode(model, src_ids, src_pad_mask, args.max_len, tgt_tokenizer, args, device, beam_size=k)
        fast_time = time.time() - start_time
        log(f"Optimized (Fast) Beam Search: {fast_time:.2f}s")
        log(f"Speedup: {slow_time / fast_time:.2f}x")
    
    log_file.close()

if __name__ == "__main__":
    run_benchmark()
