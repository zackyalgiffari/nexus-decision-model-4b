"""Compare letter logits: MLX base+adapter vs MLX fused vs Transformers fused."""
import gc
import json
import sys

import numpy as np

from nexus_decision.runtime import NexusDecision
from nexus_decision.schema import Choice

adapters, fused = sys.argv[1], sys.argv[2]
state = {"channel": "email", "body": "I was billed twice for order #4471, please refund one."}
q = Choice(instructions="Which queue?", options={"billing": "money", "shipping": "parcels", "needs_review": "other"})


def logits(nd):
    _, z, _ = nd.question_logits(state, "route", q, None, "fast")
    return np.asarray(z)


from nexus_decision.runtime_mlx import MLXBackend  # noqa: E402

res = {}
res["mlx_adapter"] = logits(NexusDecision(MLXBackend("Qwen/Qwen3.5-4B", adapter_path=adapters)))
gc.collect()
res["mlx_fused"] = logits(NexusDecision(MLXBackend(fused)))
gc.collect()
import mlx.core as mx  # noqa: E402

mx.clear_cache()
from nexus_decision.runtime_hf import HFBackend  # noqa: E402

hf = HFBackend(fused, device="cpu", dtype="float32")
print("hf class:", type(hf.model).__name__)
res["hf_fused"] = logits(NexusDecision(hf))
for k, v in res.items():
    print(k, np.round(v, 3).tolist())
d1 = np.abs(res["mlx_adapter"] - res["mlx_fused"]).max()
d2 = np.abs(res["mlx_fused"] - res["hf_fused"]).max()
print(json.dumps({"max|adapter-fused|": float(d1), "max|mlx-hf|": float(d2)}))
