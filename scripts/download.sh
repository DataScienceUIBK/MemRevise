#!/bin/bash
# Download the passage collections and the models from the Hugging Face Hub into this repository (about 8 GB).
#   pip install huggingface_hub; bash scripts/download.sh
cd "$(dirname "$0")/.." || exit 1
HF=$(ls .venv-writer/bin/hf .venv-retriever/bin/hf 2>/dev/null | head -1)
[ -n "$HF" ] || HF=$(command -v hf || command -v huggingface-cli) || { echo "install huggingface_hub first: pip install huggingface_hub"; exit 1; }
"$HF" download mohammed933/ConvRevise --repo-type dataset --include "corpus/*" --local-dir . || exit 1
for part in base writer; do                    # one pattern per call: hf reads a second pattern as a file name
  "$HF" download mohammed933/MemRevise-models --include "$part/*" --local-dir models || exit 1
done
echo "corpus/ and models/ ready; next: python -m memrevise.encode_corpus --corpus topiocqa (and coral, orquac)"
