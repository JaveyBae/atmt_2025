import os
import logging
import argparse
import time
import numpy as np
import sacrebleu
from tqdm import tqdm

import torch
import sentencepiece as spm
from torch.serialization import default_restore_location

import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from seq2seq.decode import decode, beam_search_decode
from seq2seq.data.tokenizer import BPETokenizer
from seq2seq import models, utils
from seq2seq.data.dataset import Seq2SeqDataset, BatchSampler


def decode_to_string(tokenizer, array):
    """Takes a tensor of token IDs and decodes it back into a string."""
    if torch.is_tensor(array) and array.dim() == 2:
        return '\n'.join(decode_to_string(tokenizer, t) for t in array)
    return tokenizer.Decode(array.tolist())


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--cuda', action='store_true', help='Use a GPU')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')

    # Add data arguments
    parser.add_argument('--input', required=True, help='Path to the raw text file to translate (one sentence per line)')
    parser.add_argument('--src-tokenizer', required=True, help='path to source sentencepiece tokenizer')
    parser.add_argument('--tgt-tokenizer', required=True, help='path to target sentencepiece tokenizer')
    parser.add_argument('--checkpoint-path', required=True, help='path to the model file')
    parser.add_argument('--batch-size', default=1, type=int, help='maximum number of sentences in a batch')
    parser.add_argument('--output', required=True, type=str, help='path to the output file destination')
    parser.add_argument('--max-len', default=128, type=int, help='maximum length of generated sequence')

    # Add decoder arguments
    parser.add_argument('--decoder', default='greedy', choices=['greedy', 'beam'],
                        help='Decoding strategy')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='Beam size for beam search decoding')

    # BLEU computation arguments
    parser.add_argument('--bleu', action='store_true', help='If set, compute BLEU score after translation')
    parser.add_argument('--reference', type=str, help='Path to the reference file (required if --bleu is set)')

    return parser.parse_args()


def main(args):
    """ Main translation function """
    torch.manual_seed(args.seed)
    state_dict = torch.load(args.checkpoint_path,
                            map_location=lambda s, l: default_restore_location(s, 'cpu'),
                            weights_only=False)
    args_loaded = argparse.Namespace(**{**vars(state_dict['args']), **vars(args)})
    args = args_loaded
    utils.init_logging(args)

    src_tokenizer = utils.load_tokenizer(args.src_tokenizer)
    tgt_tokenizer = utils.load_tokenizer(args.tgt_tokenizer)

    # Build model
    model = models.build_model(args, src_tokenizer, tgt_tokenizer)
    if args.cuda:
        model = model.cuda()
    model.load_state_dict(state_dict['model'])
    model.eval()
    logging.info(f'Loaded model from {args.checkpoint_path}')

    # Read input
    with open(args.input, encoding="utf-8") as f:
        src_lines = [line.strip() for line in f if line.strip()]

    # Encode
    src_encoded = [torch.tensor(src_tokenizer.Encode(line, out_type=int, add_eos=True)) for line in src_lines]
    max_seq_len = min(model.encoder.pos_embed.size(1), args.max_len)
    src_encoded = [s if len(s) <= max_seq_len else s[:max_seq_len] for s in src_encoded]

    DEVICE = 'cuda' if args.cuda else 'cpu'
    PAD = src_tokenizer.pad_id()
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()

    print(f'PAD ID: {PAD}, BOS ID: {BOS}, EOS ID: {EOS}\n'
          f'PAD token: "{src_tokenizer.IdToPiece(PAD)}", BOS token: "{tgt_tokenizer.IdToPiece(BOS)}", EOS token: "{tgt_tokenizer.IdToPiece(EOS)}"')

    # Clear output file
    if args.output:
        open(args.output, 'w', encoding='utf-8').close()

    def postprocess_ids(ids, pad, bos, eos):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if len(ids) > 0 and ids[0] == bos:
            ids = ids[1:]
        if eos in ids:
            ids = ids[:ids.index(eos)]
        return [i for i in ids if i != pad]

    def decode_sentence(tokenizer: spm.SentencePieceProcessor, sentence_ids):
        ids = postprocess_ids(sentence_ids, PAD, BOS, EOS)
        return tokenizer.Decode(ids)

    translations = []
    start_time = time.perf_counter()
    make_batch = utils.make_batch_input(device=DEVICE, pad=PAD, max_seq_len=args.max_len)

    def batch_iter(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    # ------------------------------------------
    # Translation loop
    for batch in tqdm(batch_iter(src_encoded, args.batch_size)):
        with torch.no_grad():
            batch_lengths = [len(x) for x in batch]
            max_len = max(batch_lengths)
            batch_padded = [
                torch.cat([x, torch.full((max_len - len(x),), PAD, dtype=torch.long)]) if len(x) < max_len else x
                for x in batch
            ]
            src_tokens = torch.stack(batch_padded).to(DEVICE)
            dummy_y = torch.full_like(src_tokens, fill_value=PAD)
            src_tokens, trg_in, trg_out, src_pad_mask, trg_pad_mask = make_batch(src_tokens, dummy_y)

            # Choose decoding method
            if args.decoder == "beam":
                prediction = [beam_search_decode(
                    model=model,
                    src_tokens=src_tokens,
                    src_pad_mask=src_pad_mask,
                    max_out_len=args.max_len,
                    tgt_tokenizer=tgt_tokenizer,
                    args=args,
                    device=DEVICE,
                    beam_size=args.beam_size)]
            else:
                prediction = decode(model=model,
                                    src_tokens=src_tokens,
                                    src_pad_mask=src_pad_mask,
                                    max_out_len=args.max_len,
                                    tgt_tokenizer=tgt_tokenizer,
                                    args=args,
                                    device=DEVICE)

        # Decode + write
        for sent in prediction:
            translation = decode_sentence(tgt_tokenizer, sent)
            translations.append(translation)
            if args.output:
                with open(args.output, 'a', encoding='utf-8') as out_file:
                    out_file.write(translation + '\n')

    end_time = time.perf_counter()
    logging.info(f'Wrote {len(translations)} lines to {args.output}')
    logging.info(f'Translation completed in {end_time - start_time:.2f} seconds')

    # Optional BLEU
    if getattr(args, 'bleu', False):
        with open(args.reference, encoding='utf-8') as ref_file:
            references = [line.strip() for line in ref_file if line.strip()]
        bleu = sacrebleu.corpus_bleu(translations, [references])
        print(f"BLEU score: {bleu.score:.2f}")


if __name__ == '__main__':
    args = get_args()
    if getattr(args, 'bleu', False) and not args.reference:
        raise ValueError("You must provide --reference when using --bleu.")
    main(args)
