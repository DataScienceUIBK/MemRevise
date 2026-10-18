#!/bin/bash
#SBATCH --job-name=memrevise
#SBATCH -N 1 --gres=gpu:4 --time=03:00:00
#SBATCH --output=logs/memrevise_%j.log
# One evaluation set on one 4-GPU node: memory, then search, 4 shards each.   SET=test_topiocqa sbatch scripts/slurm_example.sh
# WRITER_ENV / RETRIEVER_ENV: the two Python environments of requirements.txt.
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1; mkdir -p logs
S=${SET:?set SET}; export MEMREVISE_RETRIEVER_PYTHON=${RETRIEVER_ENV:?}/bin/python
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i ${WRITER_ENV:?}/bin/python -m memrevise.write_memory --set $S --shard $i/4 & done; wait
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i $RETRIEVER_ENV/bin/python -m memrevise.search --set $S --shard $i/4 & done; wait
$RETRIEVER_ENV/bin/python -m memrevise.evaluate --sets $S
