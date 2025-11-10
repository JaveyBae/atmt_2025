import pickle
import torch
import sentencepiece as spm

# === 路径 ===
src_pickle = "cz-en/data/prepared/test.cz"
tgt_pickle = "cz-en/data/prepared/test.en"
src_tokenizer_path = "cz-en/tokenizers/cz-bpe-8000.model"
tgt_tokenizer_path = "cz-en/tokenizers/en-bpe-8000.model"
output_src = "cz-en/data/prepared/test1.cz"
output_tgt = "cz-en/data/prepared/test1.en"

# === 加载 tokenizers ===
src_sp = spm.SentencePieceProcessor()
tgt_sp = spm.SentencePieceProcessor()
src_sp.Load(src_tokenizer_path)
tgt_sp.Load(tgt_tokenizer_path)

# === 反序列化 pickle ===
with open(src_pickle, "rb") as f:
    src_data = pickle.load(f)
with open(tgt_pickle, "rb") as f:
    tgt_data = pickle.load(f)

# === 转回文本 ===
with open(output_src, "w", encoding="utf-8") as fs, \
     open(output_tgt, "w", encoding="utf-8") as ft:
    for src, tgt in zip(src_data, tgt_data):
        src_text = src_sp.DecodeIds(list(map(int, src)))
        tgt_text = tgt_sp.DecodeIds(list(map(int, tgt)))
        fs.write(src_text.strip() + "\n")
        ft.write(tgt_text.strip() + "\n")

print(f"✅ Done! Written to:\n  {output_src}\n  {output_tgt}")
