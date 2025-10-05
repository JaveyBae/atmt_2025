#!/bin/bash
#SBATCH --job-name=translate
#SBATCH --output=out_translate.out
#SBATCH --error=err_translate.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --partition=teaching

# 加载环境
module load gpu
module load mamba
source activate atmt

# 只运行翻译
python translate.py --cuda \
  --input cz-en/data/prepared/test.cz \
  --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
  --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
  --output cz-en/output.txt \
  --max-len 300

# 计算 BLEU
python score.py \
  --reference cz-en/data/prepared/test.en \
  --translation cz-en/output.txt
