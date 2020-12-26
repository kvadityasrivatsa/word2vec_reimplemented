#!/bin/sh
#SBATCH -A research
#SBATCH -n 35
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=2048
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END

cd mikolov_was_high
#python3 word2vec.py -s 50 -w 4 -e 30 -l 0.01 -i data/sent/buffer.pkl -o sgsfSTP50_mc20 -t -1 -sl -sb -fr 1 -mc 20 -sg -sp
#python3 word2vec.py -s 50 -w 4 -e 10 -l 0.03 -i data/sent/buffer.pkl -o sgnsSTP50_mc20_nc10 -t -1 -sl -sb -fr 1 -mc 20 -ns -sg -sp -nc 10
#python3 word2vec.py -s 50 -w 4 -e 30 -l 0.02 -i data/sent/buffer.pkl -o cbsfSTP50_mc20_TEST -t -1 -sl -sb -fr 1 -mc 20 -sp
#python3 word2vec.py -s 50 -w 4 -e 30 -l 0.02 -i data/sent/buffer.pkl -o cbsfHUH -t -1 -sl -sb -fr 1 -mc 20 -sp
#python3 word2vec.py -s 50 -w 4 -e 30 -l 0.02 -i data/sent/buffer.pkl -o cbsfINC50_mc20 -t -1 -sl -sb -fr 1 -mc 20 -sp
python3 word2vec.py -s 50 -w 4 -e 30 -l 0.02 -i data/sent/buffer.pkl -o cbsfLST50_mc4 -t -1 -sl -sb -fr 1 -mc 4 -sp

