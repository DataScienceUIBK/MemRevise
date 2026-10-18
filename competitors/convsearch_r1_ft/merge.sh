#!/bin/bash
# Merge the last checkpoint of competitors/convsearch_r1_ft/train.sh (verl FSDP shards, step 39) into one Hugging Face model with the
# authors' scripts/model_merger.py, then add the tokenizer and generation files of the released model, which the merger does not write.
# CPU only, a few minutes, in the verl environment.
#
#   CONVSEARCH_R1_REPO=<clone of github.com/BeastyZ/ConvSearch-R1> bash competitors/convsearch_r1_ft/merge.sh   -> models/convsearch_r1_ft/
set -eu
ROOT=${MEMREVISE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
REPO=${CONVSEARCH_R1_REPO:?point CONVSEARCH_R1_REPO at a clone of github.com/BeastyZ/ConvSearch-R1}
CK=$ROOT/models/convsearch_r1_ft_ckpt
STEP=$(cat "$CK/latest_checkpointed_iteration.txt")
A=$CK/global_step_$STEP/actor
OUT=$ROOT/models/convsearch_r1_ft
(cd "$REPO/verl" && python3 scripts/model_merger.py --local_dir "$A")
rm -rf "$OUT"
cp -r "$A/huggingface" "$OUT"
REL=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('BeastyZ/Qwen2.5-3B-ConvSearch-R1-TopiOCQA'))")
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json generation_config.json; do
    [ -s "$OUT/$f" ] || cp -L "$REL/$f" "$OUT/"
done
echo "step $STEP -> $OUT"
