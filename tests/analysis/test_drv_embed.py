import json

import numpy as np
import torch

from scripts.analysis import drv_embed
from scripts.read_token_store import shard_meta, write_store
from src.harness.learners import build_learner
from src.repstore.store import TOKEN_SEQ, RepSpec, ShardedRepSplit

D, R = 8, 4


def drv_cell(path, d, spec="transformer:d16,l1,f32,h2"):
    torch.manual_seed(0)
    model = build_learner(spec, d, t_max=512, dropout=0.1)
    path.mkdir()
    torch.save(model.state_dict(), path / "model.pt")
    (path / "results.json").write_text(json.dumps({
        "learner": spec, "in_domain": {"auroc": 0.5},
        "protocol": {"t_max": 512, "dropout": 0.1, "rescale": "none"}}))
    return model.eval()


def test_embedding_is_what_the_head_reads_and_logits_match(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    model = drv_cell(tmp_path / "cell", D)
    items = rng.normal(size=(5, R, D)).astype(np.float32)
    meta = [{"uid": f"u{k}", "global_index": k} for k in range(5)]
    write_store(tmp_path / "rt" / "val_5k" / "shard_00_000", items.reshape(-1, D),
                np.full(5, R), np.arange(5) % 2, shard_meta(meta, R),
                RepSpec(name="rt_layers", kind=TOKEN_SEQ, dim=D, layer=-1, backbone="t", readout="t"))
    monkeypatch.setattr("sys.argv", ["x", "--cell_dir", str(tmp_path / "cell"), "--layers_root",
                                     str(tmp_path / "rt"), "--out_root", str(tmp_path / "z")])
    drv_embed.main()

    out = ShardedRepSplit(tmp_path / "z" / "val_5k")
    assert out.is_vector and [m["uid"] for m in out.meta()] == [m["uid"] for m in meta]
    x = torch.from_numpy(items[:, 1:])  # row 0 is the boundary slot
    with torch.no_grad():
        h = model.enc(model.proj(x) + model.pos[:R - 1].unsqueeze(0))
        z = h.mean(1)
        logit = model.head(z).squeeze(-1)
    Z = np.stack([out.item(k)[0] for k in range(5)])
    assert np.allclose(Z, z.numpy(), atol=1e-5)
    L = np.load(tmp_path / "z" / "val_5k" / "shard_00_000" / "logits.npy")
    assert np.allclose(L, logit.numpy(), atol=1e-5)
