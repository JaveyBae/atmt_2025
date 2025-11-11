import os
import logging
import argparse
import time
import torch
import sentencepiece as spm
from torch.serialization import default_restore_location
from tqdm import tqdm
import sacrebleu

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq import models, utils
from seq2seq.decode import decode

def get_args():
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input', required=True)
    parser.add_argument('--src-tokenizer', required=True)
    parser.add_argument('--tgt-tokenizer', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--bleu', action='store_true')
    parser.add_argument('--reference', type=str)
    return parser.parse_args()

def postprocess_ids(ids, pad, bos, eos):
    """Remove BOS, truncate at EOS, remove PADs."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if len(ids) > 0 and ids[0] == bos:
        ids = ids[1:]
    if eos in ids:
        ids = ids[:ids.index(eos)]
    ids = [i for i in ids if i != pad]
    return ids

def decode_sentence(tokenizer: spm.SentencePieceProcessor, sentence_ids, PAD, BOS, EOS):
    ids = postprocess_ids(sentence_ids, PAD, BOS, EOS)
    if len(ids) == 0:
        return ""
    return tokenizer.Decode(ids)

def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def main(args):
    torch.manual_seed(args.seed)
    DEVICE = 'cuda' if args.cuda else 'cpu'

    # Load checkpoint
    state_dict = torch.load(args.checkpoint_path, map_location=lambda s,l: default_restore_location(s,'cpu'), weights_only=False)
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)

    # Load tokenizers
    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)
    PAD, BOS, EOS = src_tokenizer.pad_id(), tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id()
    print(f"PAD: {PAD}, BOS: {BOS}, EOS: {EOS}")

    # Build model
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    model.load_state_dict(state_dict['model'])
    if args.cuda:
        model = model.cuda()
    model.eval()
    logging.info(f"Loaded model from {args.checkpoint_path}")

    # Read and encode input
    with open(args.input, encoding='utf-8') as f:
        src_lines = [line.strip() for line in f if line.strip()]
    src_encoded = [torch.tensor(src_tokenizer.Encode(line, out_type=int, add_eos=True)) for line in src_lines]
    max_seq_len = min(getattr(model.encoder.rope, 'max_seq_len', args.max_len), args.max_len)
    src_encoded = [s[:max_seq_len] if len(s) > max_seq_len else s for s in src_encoded]

    # Clear output file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('')

    # Prepare batch utils
    make_batch = utils.make_batch_input(device=DEVICE, pad=PAD, max_seq_len=args.max_len)

    translations = []
    start_time = time.perf_counter()

    # Translation loop
    for batch in tqdm(batch_iter(src_encoded, args.batch_size)):
        with torch.no_grad():
            max_len = max(len(x) for x in batch)
            batch_padded = [torch.cat([x, torch.full((max_len-len(x),), PAD, dtype=torch.long)]) if len(x)<max_len else x for x in batch]
            src_tokens = torch.stack(batch_padded).to(DEVICE)
            dummy_y = torch.full_like(src_tokens, PAD)
            src_tokens, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch(src_tokens, dummy_y)

            prediction = decode(model=model,
                                src_tokens=src_tokens,
                                src_pad_mask=src_pad_mask,
                                max_out_len=args.max_len,
                                tgt_tokenizer=tgt_tokenizer,
                                args=args,
                                device=DEVICE)

        for sent in prediction:
            translation = decode_sentence(tgt_tokenizer, sent, PAD, BOS, EOS)
            translations.append(translation)
            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(translation+'\n')

    logging.info(f"Wrote {len(translations)} lines to {args.output}")
    end_time = time.perf_counter()
    logging.info(f"Translation completed in {end_time - start_time:.2f}s")

    # Compute BLEU
    if getattr(args, 'bleu', False):
        if not args.reference:
            raise ValueError("Must provide --reference when using --bleu")
        with open(args.reference, encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()]
        if len(references) != len(translations):
            raise ValueError(f"Reference ({len(references)}) and hypothesis ({len(translations)}) line counts do not match.")
        bleu = sacrebleu.corpus_bleu(translations, [references])
        print(f"BLEU score: {bleu.score:.2f}")

if __name__ == '__main__':
    args = get_args()
    main(args)
