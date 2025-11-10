#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --output=out_task2_beam.out

# === environment ===
module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# === path ===
DATA_DIR=cz-en/data/prepared
TOKENIZER_DIR=cz-en/tokenizers
CKPT=cz-en/checkpoints/checkpoint_best.pt
OUTPUT_DIR=toy_example/task2_outputs
mkdir -p $OUTPUT_DIR

# === 1️⃣ Greedy baseline ===
echo "=== Greedy decoding (baseline) ==="
python translate.py \
    --cuda \
    --input $DATA_DIR/test1.cz \
    --src-tokenizer $TOKENIZER_DIR/cz-bpe-8000.model \
    --tgt-tokenizer $TOKENIZER_DIR/en-bpe-8000.model \
    --checkpoint-path $CKPT \
    --output $OUTPUT_DIR/output_greedy.txt \
    --decoder greedy \
    --max-len 300 \
    --bleu \
    --reference $DATA_DIR/test1.en \
    > $OUTPUT_DIR/bleu_greedy.log

# === 2️⃣ Beam Search (beam_size=3) ===
echo "=== Beam Search (beam_size=3) ==="
python translate.py \
    --cuda \
    --input $DATA_DIR/test1.cz \
    --src-tokenizer $TOKENIZER_DIR/cz-bpe-8000.model \
    --tgt-tokenizer $TOKENIZER_DIR/en-bpe-8000.model \
    --checkpoint-path $CKPT \
    --output $OUTPUT_DIR/output_beam3.txt \
    --decoder beam \
    --beam-size 3 \
    --max-len 300 \
    --bleu \
    --reference $DATA_DIR/test1.en \
    > $OUTPUT_DIR/bleu_beam3.log

# === 3️⃣ Beam Search (beam_size=5) ===
echo "=== Beam Search (beam_size=5) ==="
python translate.py \
    --cuda \
    --input $DATA_DIR/test1.cz \
    --src-tokenizer $TOKENIZER_DIR/cz-bpe-8000.model \
    --tgt-tokenizer $TOKENIZER_DIR/en-bpe-8000.model \
    --checkpoint-path $CKPT \
    --output $OUTPUT_DIR/output_beam5.txt \
    --decoder beam \
    --beam-size 5 \
    --max-len 300 \
    --bleu \
    --reference $DATA_DIR/test1.en \
    > $OUTPUT_DIR/bleu_beam5.log

# === 4️⃣ Beam Search (beam_size=10) ===
echo "=== Beam Search (beam_size=10) ==="
python translate.py \
    --cuda \
    --input $DATA_DIR/test1.cz \
    --src-tokenizer $TOKENIZER_DIR/cz-bpe-8000.model \
    --tgt-tokenizer $TOKENIZER_DIR/en-bpe-8000.model \
    --checkpoint-path $CKPT \
    --output $OUTPUT_DIR/output_beam10.txt \
    --decoder beam \
    --beam-size 10 \
    --max-len 300 \
    --bleu \
    --reference $DATA_DIR/test1.en \
    > $OUTPUT_DIR/bleu_beam10.log

# ===  BLEU  ===
echo "=== All results saved in $OUTPUT_DIR ==="
grep "BLEU score" $OUTPUT_DIR/*.log || echo "o BLEU results found."
