"""Shared GPU runtime for parametric_retrieval_access_v1 causal experiments.

Model/tokenizer loading, stored-hidden-state lookup (HSStore), residual-stream
intervention hooks (patch at one position / steer from one position onward),
and batched measurement: greedy continuation, teacher-forced answer logP,
decision-token entropy and candidate rank.

Layer convention: hs_idx k = output of model.model.layers[k-1] (resid_post).
Interventions hook that module and edit its output hidden states.

Padding convention: generation uses LEFT padding (position = mx-1 is the
final prompt token for every sample); scoring uses RIGHT padding with
attention mask (position = prompt_len-1 per sample).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.parametric_retrieval.prga_generate import (  # noqa: E402
    encode_prompt,
)


def load_model_and_tok(model_name: str, local_files_only: bool,
                       dtype_str: str = "bfloat16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name,
                                        local_files_only=local_files_only)
    dtype = getattr(torch, dtype_str)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=local_files_only, torch_dtype=dtype)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tok, torch.device(device)


class HSStore:
    """Row lookup into the stage-3 extraction: (instance_id, position_name)
    -> stored residual vector at a given hs_idx. Loads whole layer arrays
    lazily (fp16, ~1 GB each)."""

    def __init__(self, out_dir: Path):
        self.hs_dir = out_dir / "hidden_states_v1"
        meta = pd.read_parquet(self.hs_dir / "hs_meta.parquet")
        self.meta = meta
        self.row_of = {(r.instance_id, r.position_name): i
                       for i, r in enumerate(meta.itertuples())}
        self._layers: dict[int, np.ndarray] = {}

    def layer(self, hs_idx: int) -> np.ndarray:
        if hs_idx not in self._layers:
            from safetensors.numpy import load_file
            self._layers[hs_idx] = load_file(
                str(self.hs_dir / f"layer_{hs_idx:02d}.safetensors"))["h"]
        return self._layers[hs_idx]

    def vec(self, instance_id: str, position_name: str,
            hs_idx: int) -> np.ndarray | None:
        i = self.row_of.get((instance_id, position_name))
        return None if i is None else self.layer(hs_idx)[i]


class ResidualEdit:
    """Context-managed forward hook editing residual outputs of one block.

    mode='patch': out[b, idx[b]] += alpha * (vec[b] - out[b, idx[b]])
                  applied only on prefill passes (seq_len > 1).
    mode='add':   out[b, idx[b]] += alpha * vec[b] on prefill only. Use for
                  partial / masked injections where vec is a precomputed delta
                  (e.g. a subset of coordinates of h_donor - h_recip, or a
                  neuron-subset reconstruction W_down[:, S] @ dg[S]).
    mode='steer': out[b, idx[b]:] += vec[b] on prefill AND out[b, -1] +=
                  vec[b] on every decode step (seq_len == 1).
    idx are indices into the current (padded) sequence. Samples with idx < 0
    are left untouched (lets a batch mix conditions).
    """

    def __init__(self, model, hs_idx: int, mode: str):
        if mode not in ("patch", "add", "steer"):
            raise ValueError(mode)
        self.block = model.model.layers[hs_idx - 1]
        self.mode = mode
        self.idx = None
        self.vecs = None
        self.alpha = 1.0
        self._handle = None

    def set(self, idx, vecs, alpha: float = 1.0):
        self.idx = idx
        self.vecs = vecs
        self.alpha = alpha

    def _hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.idx is None:
            return output
        seq_len = h.shape[1]
        for b, i in enumerate(self.idx):
            if i is None or i < 0:
                continue
            v = self.vecs[b].to(h.dtype)
            if self.mode == "patch":
                if seq_len > 1:
                    h[b, i] = h[b, i] + self.alpha * (v - h[b, i])
            elif self.mode == "add":
                if seq_len > 1:
                    h[b, i] = h[b, i] + self.alpha * v
            else:
                if seq_len > 1:
                    h[b, i:] = h[b, i:] + v
                else:
                    h[b, -1] = h[b, -1] + v
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    def __enter__(self):
        self._handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self.idx = None
        return False


class ComponentEdit:
    """Context-managed forward hook editing ONE sub-block's output (the
    attention contribution or the MLP contribution added to the residual
    stream) of one decoder layer, at one token position.

    Qwen2DecoderLayer adds self_attn(...) then mlp(...) into the residual, so
    hooking those submodules edits exactly the additive contribution of that
    component. kind='attn' -> layers[layer].self_attn (tuple output, hidden is
    element 0); kind='mlp' -> layers[layer].mlp (plain tensor output).

    Same edit rule and interface as ResidualEdit (patch mode only, prefill
    only) so it is a drop-in for generate_with_edit / score_with_edit:
        out[b, idx[b]] += alpha * (vec[b] - out[b, idx[b]]).
    idx < 0 leaves a sample untouched (lets a batch mix conditions)."""

    def __init__(self, model, layer: int, kind: str):
        if kind not in ("attn", "mlp"):
            raise ValueError(kind)
        blk = model.model.layers[layer]
        self.block = blk.self_attn if kind == "attn" else blk.mlp
        self.kind = kind
        self.idx = None
        self.vecs = None
        self.alpha = 1.0
        self._handle = None

    def set(self, idx, vecs, alpha: float = 1.0):
        self.idx = idx
        self.vecs = vecs
        self.alpha = alpha

    def _hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.idx is None:
            return output
        if h.shape[1] > 1:  # prefill only
            for b, i in enumerate(self.idx):
                if i is None or i < 0:
                    continue
                v = self.vecs[b].to(h.dtype)
                h[b, i] = h[b, i] + self.alpha * (v - h[b, i])
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h

    def __enter__(self):
        self._handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        self.idx = None
        return False


class ComponentStore:
    """Row lookup into the component extraction (component_states_v1):
    instance_id -> stored attn / mlp contribution vector at the final prompt
    token, per decoder layer. Loads whole layer arrays lazily (fp16)."""

    def __init__(self, out_dir: Path):
        self.hs_dir = out_dir / "component_states_v1"
        self.meta = pd.read_parquet(self.hs_dir / "comp_meta.parquet")
        self.row_of = {r.instance_id: i
                       for i, r in enumerate(self.meta.itertuples())}
        self._layers: dict[tuple, np.ndarray] = {}

    def layer(self, kind: str, layer: int) -> np.ndarray:
        key = (kind, layer)
        if key not in self._layers:
            from safetensors.numpy import load_file
            self._layers[key] = load_file(
                str(self.hs_dir / f"{kind}_L{layer:02d}.safetensors"))["h"]
        return self._layers[key]

    def vec(self, instance_id: str, kind: str, layer: int) -> np.ndarray | None:
        i = self.row_of.get(instance_id)
        return None if i is None else self.layer(kind, layer)[i]


class ClampNeuron:
    """Context-managed forward-pre-hook that clamps a set of MLP intermediate
    neurons to fixed values on EVERY token and every forward pass (prefill and
    each decode step), i.e. Golden-Gate-style persistent feature steering, not
    a one-position patch. Hooks the layer's down_proj input g (mlp_out =
    W_down @ g), so g[:, :, nid] := value for each (nid, value).

    All samples in the batch get the same clamp; drive one neuron set per
    batch. set(nids, vals) before the generate/forward call."""

    def __init__(self, model, layer: int):
        self.block = model.model.layers[layer].mlp.down_proj
        self.nids: list[int] = []
        self.vals: list[float] = []
        self._handle = None

    def set(self, nids, vals):
        self.nids = list(nids)
        self.vals = list(vals)

    def _hook(self, module, args):
        g = args[0]
        if self.nids:
            for nid, val in zip(self.nids, self.vals):
                g[:, :, nid] = val
        return (g,) + tuple(args[1:])

    def __enter__(self):
        self._handle = self.block.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False


def greedy_generate(model, tok, device, prompt_ids: list[list[int]],
                    max_new_tokens: int = 40, clamp=None) -> list[str]:
    """LEFT-padded greedy generation, optionally under a persistent clamp
    context (ClampNeuron). Returns decoded continuations."""
    import torch
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    mx = max(len(x) for x in prompt_ids)
    inp = torch.tensor([[pad] * (mx - len(x)) + x for x in prompt_ids],
                       dtype=torch.long, device=device)
    att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                        for x in prompt_ids], dtype=torch.long, device=device)
    ctx = clamp if clamp is not None else _null_ctx()
    with torch.no_grad(), ctx:
        seqs = model.generate(inp, attention_mask=att, do_sample=False,
                              max_new_tokens=max_new_tokens, pad_token_id=pad)
    return [tok.decode(seqs[b, mx:].tolist(), skip_special_tokens=True)
            for b in range(len(prompt_ids))]


class NeuronStore:
    """Row lookup into the neuron extraction (neuron_states_v1): instance_id ->
    stored MLP intermediate activation g = silu(gate(x)) * up(x) at the final
    prompt token, per decoder layer. g is the (intermediate_size,) pre-down_proj
    vector, so mlp_out = down_proj.weight @ g. Loads lazily (fp16)."""

    def __init__(self, out_dir: Path):
        self.hs_dir = out_dir / "neuron_states_v1"
        self.meta = pd.read_parquet(self.hs_dir / "neuron_meta.parquet")
        self.row_of = {r.instance_id: i
                       for i, r in enumerate(self.meta.itertuples())}
        self._layers: dict[int, np.ndarray] = {}

    def layer(self, layer: int) -> np.ndarray:
        if layer not in self._layers:
            from safetensors.numpy import load_file
            self._layers[layer] = load_file(
                str(self.hs_dir / f"g_L{layer:02d}.safetensors"))["h"]
        return self._layers[layer]

    def vec(self, instance_id: str, layer: int) -> np.ndarray | None:
        i = self.row_of.get(instance_id)
        return None if i is None else self.layer(layer)[i]


def capture_readout(model, tok, device, prompt_ids: list[list[int]],
                    edit, edit_vecs, edit_alpha, readout_hs_idx: int):
    """RIGHT-padded forward, optionally under an edit, returning the residual
    hidden state at readout_hs_idx and the final prompt token for each sample
    (n, hidden) as float32 numpy. Used to see where an intervention lands a
    recipient in representation space downstream of the patched layer."""
    import torch
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    mx = max(len(x) for x in prompt_ids)
    inp = torch.tensor([x + [pad] * (mx - len(x)) for x in prompt_ids],
                       dtype=torch.long, device=device)
    att = torch.tensor([[1] * len(x) + [0] * (mx - len(x))
                        for x in prompt_ids], dtype=torch.long, device=device)
    ctx = edit if edit is not None else _null_ctx()
    with torch.no_grad(), ctx:
        if edit is not None:
            idx = [len(p) - 1 if v is not None else -1
                   for p, v in zip(prompt_ids, edit_vecs)]
            vecs = [torch.zeros(model.config.hidden_size, device=device)
                    if v is None else torch.as_tensor(v, device=device)
                    for v in edit_vecs]
            edit.set(idx, vecs, edit_alpha)
        hs = model(inp, attention_mask=att, output_hidden_states=True,
                   use_cache=False).hidden_states[readout_hs_idx]
    out = np.empty((len(prompt_ids), model.config.hidden_size),
                   dtype=np.float32)
    for b, p in enumerate(prompt_ids):
        out[b] = hs[b, len(p) - 1].float().cpu().numpy()
    return out


def first_token_ids(tok, answers: list[str]) -> list[int]:
    out = []
    for a in answers:
        ids = tok(" " + str(a), add_special_tokens=False)["input_ids"]
        out.append(int(ids[0]) if ids else -1)
    return out


def generate_with_edit(model, tok, device, prompt_ids: list[list[int]],
                       edit: ResidualEdit | None, edit_vecs, edit_alpha,
                       max_new_tokens: int = 16) -> list[str]:
    """LEFT-padded greedy generation; the edit targets the final prompt
    token (index mx-1) of every sample whose edit vector is not None."""
    import torch
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    mx = max(len(x) for x in prompt_ids)
    inp = torch.tensor([[pad] * (mx - len(x)) + x for x in prompt_ids],
                       dtype=torch.long, device=device)
    att = torch.tensor([[0] * (mx - len(x)) + [1] * len(x)
                        for x in prompt_ids], dtype=torch.long, device=device)
    ctx = edit if edit is not None else _null_ctx()
    with torch.no_grad(), ctx:
        if edit is not None:
            idx = [mx - 1 if v is not None else -1 for v in edit_vecs]
            vecs = [torch.zeros(model.config.hidden_size, device=device)
                    if v is None else torch.as_tensor(v, device=device)
                    for v in edit_vecs]
            edit.set(idx, vecs, edit_alpha)
        seqs = model.generate(inp, attention_mask=att, do_sample=False,
                              max_new_tokens=max_new_tokens,
                              pad_token_id=pad)
    texts = []
    for b in range(len(prompt_ids)):
        gen = seqs[b, mx:].tolist()
        texts.append(tok.decode(gen, skip_special_tokens=True))
    return texts


def score_with_edit(model, tok, device, prompt_ids: list[list[int]],
                    answers: list[str], edit: ResidualEdit | None,
                    edit_vecs, edit_alpha,
                    cand_ids_rows: list[list[int]] | None = None):
    """RIGHT-padded teacher-forced scoring of ' <answer>' after each prompt.

    Returns per sample: logp_answer (sum over answer tokens), entropy at the
    decision token (nats), gold_rank among cand_ids_rows[b] first tokens
    (rank of cand_ids_rows[b][0]; None when not given).
    """
    import torch
    import torch.nn.functional as tF
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ans_ids = [tok(" " + str(a), add_special_tokens=False)["input_ids"]
               for a in answers]
    seqs = [p + a for p, a in zip(prompt_ids, ans_ids)]
    mx = max(len(s) for s in seqs)
    inp = torch.tensor([s + [pad] * (mx - len(s)) for s in seqs],
                       dtype=torch.long, device=device)
    att = torch.tensor([[1] * len(s) + [0] * (mx - len(s)) for s in seqs],
                       dtype=torch.long, device=device)
    ctx = edit if edit is not None else _null_ctx()
    with torch.no_grad(), ctx:
        if edit is not None:
            idx = [len(p) - 1 if v is not None else -1
                   for p, v in zip(prompt_ids, edit_vecs)]
            vecs = [torch.zeros(model.config.hidden_size, device=device)
                    if v is None else torch.as_tensor(v, device=device)
                    for v in edit_vecs]
            edit.set(idx, vecs, edit_alpha)
        logits = model(inp, attention_mask=att, use_cache=False).logits
    out = []
    for b, (p, a) in enumerate(zip(prompt_ids, ans_ids)):
        lp = 0.0
        logprobs = tF.log_softmax(logits[b].float(), dim=-1)
        for j, t in enumerate(a):
            lp += float(logprobs[len(p) - 1 + j, t])
        dec = logprobs[len(p) - 1]
        entropy = float(-(dec.exp() * dec).sum())
        rank = None
        if cand_ids_rows is not None and cand_ids_rows[b]:
            cs = cand_ids_rows[b]
            vals = dec[torch.tensor(cs, device=device)]
            rank = int(1 + (vals[1:] > vals[0]).sum())
        out.append({"logp_answer": lp, "entropy": entropy,
                    "gold_rank": rank})
    return out


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def load_candidate_table(out_dir: Path, tok) -> dict:
    """(fact_id, direction) -> {'gold', 'negatives', 'first_ids'} with
    first-token collisions against the gold removed (as in the logit lens)."""
    table = {}
    for c in json.loads((out_dir / "candidates.json").read_text()):
        ids = first_token_ids(tok, [c["gold"]] + c["negatives"])
        keep, seen = [ids[0]], {ids[0]}
        negs = []
        for n, t in zip(c["negatives"], ids[1:]):
            if t >= 0 and t not in seen:
                seen.add(t)
                keep.append(t)
                negs.append(n)
        table[(str(c["fact_id"]), c["direction"])] = {
            "gold": c["gold"], "negatives": negs, "first_ids": keep}
    return table


def render_prompt_ids(tok, meta: pd.DataFrame) -> dict[str, list[int]]:
    """instance_id -> prompt token ids (chat-template text tokenization)."""
    return {r.instance_id: encode_prompt(tok, r.user_message)
            for r in meta.itertuples()}
