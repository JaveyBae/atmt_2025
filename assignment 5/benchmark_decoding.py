import time
import torch
import logging
import torch.nn as nn
from seq2seq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder
from seq2seq.decode import decode as slow_decode, beam_search_decode as slow_beam_decode
from seq2seq.decode_optimized_A5T3 import decode as fast_decode, beam_search_decode as fast_beam_decode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("benchmark_results.log"),
        logging.StreamHandler()
    ]
)

class DummyTokenizer:
    def bos_id(self): return 1
    def eos_id(self): return 2
    def pad_id(self): return 3
    def GetPieceSize(self): return 8000

def load_dummy_model(device):
    # Create a dummy model similar to the assignment config
    tokenizer = DummyTokenizer()
    
    encoder = TransformerEncoder(
        src_tokenizer=tokenizer,
        dim_embed=256,
        dropout=0.1,
        max_seq_len=1000,
        n_attention_heads=4,
        dim_ff=1024,
        pretrained_embedding=None,
        n_encoder_layers=3
    )
    
    decoder = TransformerDecoder(
        tgt_tokenizer=tokenizer,
        dim_embed=256,
        n_attention_heads=4,
        dropout=0.1,
        max_seq_len=1000,
        n_decoder_layers=3,
        dim_ff=1024,
        pretrained_embedding=None,
        use_cuda=(device.type == 'cuda')
    )
    
    model = TransformerModel(encoder, decoder).to(device)
    model.eval()
    return model, tokenizer

def load_dummy_data(device, num_sentences=100, seq_len=50):
    # Create dummy data, slightly longer to make Encoder overhead more obvious
    src_tokens = torch.randint(10, 8000, (num_sentences, seq_len)).to(device)
    src_pad_mask = (src_tokens == 0).unsqueeze(1).unsqueeze(2) # Assume 0 is pad
    return src_tokens, src_pad_mask

def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    model, tokenizer = load_dummy_model(device)
    
    # Increase sentence count for stable results
    num_sentences = 30
    # Increase source sequence length to make Encoder re-computation cost higher
    seq_len = 50 
    src_tokens_batch, src_pad_mask_batch = load_dummy_data(device, num_sentences, seq_len)
    
    logging.info(f"Running benchmark on {num_sentences} sentences with length {seq_len}...")
    logging.info("Note: 'Slow' versions re-run the Encoder at every step.")
    logging.info("Note: 'Fast' versions run the Encoder once per sentence.")

    # --- Greedy Decoding ---
    logging.info("\n--- Greedy Decoding ---")
    
    # Slow
    start = time.time()
    for i in range(num_sentences):
        slow_decode(model, src_tokens_batch[i:i+1], src_pad_mask_batch[i:i+1], 50, tokenizer, None, device)
    slow_time = time.time() - start
    logging.info(f"Original (Slow) Greedy: {slow_time:.2f}s")

    # Fast
    start = time.time()
    for i in range(num_sentences):
        fast_decode(model, src_tokens_batch[i:i+1], src_pad_mask_batch[i:i+1], 50, tokenizer, None, device)
    fast_time = time.time() - start
    logging.info(f"Optimized (Fast) Greedy: {fast_time:.2f}s")
    logging.info(f"Speedup: {slow_time / fast_time:.2f}x")

    # --- Beam Search ---
    for k in [1, 3, 5]:
        logging.info(f"\n--- Beam Search (k={k}) ---")
        
        # Slow
        start = time.time()
        for i in range(num_sentences):
            slow_beam_decode(model, src_tokens_batch[i:i+1], src_pad_mask_batch[i:i+1], 50, tokenizer, None, device, beam_size=k)
        slow_time = time.time() - start
        logging.info(f"Original (Slow) Beam Search: {slow_time:.2f}s")

        # Fast
        start = time.time()
        for i in range(num_sentences):
            fast_beam_decode(model, src_tokens_batch[i:i+1], src_pad_mask_batch[i:i+1], 50, tokenizer, None, device, beam_size=k)
        fast_time = time.time() - start
        logging.info(f"Optimized (Fast) Beam Search: {fast_time:.2f}s")
        logging.info(f"Speedup: {slow_time / fast_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
