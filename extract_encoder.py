"""
Extract the embedding_net (encoder) from a saved posterior.pt and save it as
a standalone state_dict checkpoint.

Usage:
    python extract_encoder.py \
        outputs/exp_cnn4e64-ae-reduced_maf5_freeze-maf_1M/posterior.pt \
        outputs/exp_cnn4e64-ae-reduced_maf5_freeze-maf_1M/enc_maf_joint.pt
"""

import sys
import torch
from pathlib import Path

if len(sys.argv) != 3:
    print(__doc__)
    sys.exit(1)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

posterior = torch.load(src, map_location="cpu", weights_only=False)
state = posterior.posterior_estimator.embedding_net.state_dict()
torch.save(state, dst)
print(f"Saved encoder ({len(state)} tensors) → {dst}")
