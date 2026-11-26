"""Greedy DSpark for the Inkling AMD runner, using the published three-layer MTP checkpoint."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def rms(x, weight):
    f = x.float()
    return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6) * weight.float()).bfloat16()


class Drafter:
    def __init__(self, model, directory, prompt_ids):
        self.w = load_file(str(directory / "mtp.safetensors"), device="cuda:0")
        self.alpha = self.w["fuse_weight"].float().softmax(-1).bfloat16().float()
        self.embed = model.get_input_embeddings()
        ranks = Path(__file__).resolve().parents[2] / "native/src/kernels/draft/draft_shortlist.bin"
        ranked = np.fromfile(ranks, dtype="<u4")[:32768].astype(np.int64)
        vocab = model.lm_head.weight.size(0)
        ids = np.unique(np.concatenate([ranked, np.arange(199990, vocab), prompt_ids]))
        self.ids = torch.tensor(ids[ids < vocab], device="cuda")
        self.head = model.lm_head.weight[self.ids].bfloat16()
        hw, hb = (self.w["markov_head.hidden_out." + name].float() for name in ("weight", "bias"))
        self.fold = (self.head.float() @ hw).bfloat16()
        self.bias = (self.head.float() @ hb).bfloat16()
        self.freq = 1e6 ** (-torch.arange(0, 128, 2, device="cuda").float() / 128)
        self.length = 0
        self.context = [None] * 3

    def rope(self, x, pos, weight):
        x = rms(x, weight).float()
        a, b = x.chunk(2, dim=-1)
        angles = (pos[:, None] * self.freq)[..., None, :]
        c, s = angles.cos(), angles.sin()
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1).bfloat16()

    def commit(self, taps):
        n = taps.size(0)
        pos = torch.arange(self.length, self.length + n, device=taps.device)
        mix = self.alpha @ taps.float()
        for layer in range(3):
            prefix = f"layers.{layer}."
            x = rms(mix[:, layer], self.w[prefix + "dflare_norm.weight"])
            attn = prefix + "self_attn."
            k = F.linear(x, self.w[attn + "k_proj_ctx.weight"]).view(n, 8, 128)
            k = self.rope(k, pos, self.w[attn + "k_norm_ctx.weight"])
            v = F.linear(x, self.w[attn + "v_proj_ctx.weight"]).view(n, 8, 128)
            cap = 65536 if layer == 1 else 1024
            if self.context[layer] is not None:
                oldk, oldv = self.context[layer]
                k, v = torch.cat((oldk, k)), torch.cat((oldv, v))
            self.context[layer] = (k[-cap:].contiguous(), v[-cap:].contiguous())
        self.feature = rms(mix[-1, 3], self.w["markov_head.dflare_norm.weight"])
        self.length += n

    def propose(self, anchor, count):
        anchor_id = torch.tensor([anchor], device="cuda")
        x = torch.cat((self.embed(anchor_id).bfloat16(), self.w["mask_embed"].expand(7, -1)))
        pos = torch.arange(self.length, self.length + 8, device=x.device)
        for layer in range(3):
            p = f"layers.{layer}."
            a = p + "self_attn."
            y = rms(x, self.w[p + "input_layernorm.weight"])
            q = self.rope(
                F.linear(y, self.w[a + "q_proj.weight"]).view(8, 24, 128), pos, self.w[a + "q_norm.weight"]
            )
            k = self.rope(
                F.linear(y, self.w[a + "k_proj.weight"]).view(8, 8, 128), pos, self.w[a + "k_norm.weight"]
            )
            v = F.linear(y, self.w[a + "v_proj.weight"]).view(8, 8, 128)
            oldk, oldv = self.context[layer]
            keys, values = (torch.cat((old, new)).transpose(0, 1) for old, new in ((oldk, k), (oldv, v)))
            out = F.scaled_dot_product_attention(q.transpose(0, 1), keys, values, enable_gqa=True)
            x = x + F.linear(out.transpose(0, 1).reshape(8, -1), self.w[a + "o_proj.weight"])
            y = rms(x, self.w[p + "post_attention_layernorm.weight"])
            gate = F.linear(y, self.w[p + "mlp.gate_proj.weight"])
            up = F.linear(y, self.w[p + "mlp.up_proj.weight"])
            x = x + F.linear(F.silu(gate) * up, self.w[p + "mlp.down_proj.weight"])
        h = rms(x[1 : count + 1], self.w["norm.weight"])
        vals, indices = F.linear(h, self.head).topk(1024, dim=-1)
        m = "markov_head."
        state = h.new_zeros(256)
        prev = anchor_id[0]
        proposals = []
        for i in range(count):
            z = torch.cat((state, self.w[m + "markov_w1.weight"][prev], h[i], self.feature))
            raw = F.linear(z, self.w[m + "rnn_proj.weight"]) + self.w[m + "rnn_proj.bias"]
            gate, cell, latent = raw.chunk(3)
            gate = gate.sigmoid()
            state = gate * state + (1 - gate) * cell.tanh()
            subset = indices[i]
            correction = F.linear(latent.tanh(), self.fold[subset]) + self.bias[subset]
            score = (30 * ((vals[i] + correction).float() / 30).tanh()).bfloat16()
            prev = self.ids[subset[score.argmax()]]
            proposals.append(prev)
        return torch.stack(proposals).tolist()


def generate(model, tokens, image, limit, ends, clock, directory, visits, depth):
    draft = Drafter(model, directory, tokens.input_ids[0].tolist())
    taps = []

    def entering(module, args):
        if visits.current in (5, 23, 41):
            taps.append(args[0][0].bfloat16())

    def final(module, args, output):
        taps.append(output[0].bfloat16())

    text = model.model.language_model
    handles = [layer.register_forward_pre_hook(entering) for layer in text.layers]
    handles.append(text.norm.register_forward_hook(final))
    try:
        first = model(**tokens, **image, use_cache=True, logits_to_keep=1)
        cache = first.cache
        anchor = int(first.logits[0, -1].argmax())
        result = [anchor]
        clock.put(None)
        clock.put(None)
        if limit > 1 and anchor not in ends:
            # Bound the temporary fusion tensors on a long prefill.
            for offset in range(0, tokens.input_ids.size(1), 512):
                draft.commit(torch.stack([t[offset : offset + 512] for t in taps], dim=1))
            cache.activate_past_recording()
        proposed = accepted = rounds = 0
        while len(result) < limit and anchor not in ends:
            count = min(depth, limit - len(result) - 1)
            guesses = draft.propose(anchor, count) if count else []
            ids = tokens.input_ids.new_tensor([[anchor, *guesses]])
            taps.clear()
            output = model(input_ids=ids, cache=cache, use_cache=True)
            targets = output.logits[0].argmax(-1).tolist()
            keep = 0
            while keep < count and guesses[keep] == targets[keep]:
                keep += 1
            proposed += count
            accepted += keep
            rounds += 1
            for token in targets[: keep + 1]:
                result.append(token)
                clock.put(None)
                if token in ends:
                    break
            anchor = result[-1]
            if len(result) < limit and anchor not in ends:
                cache.crop(keep - count)
                draft.commit(torch.stack(taps, dim=1)[: keep + 1])
        stats = {"verification_rounds": rounds, "proposed": proposed, "accepted": accepted}
        return torch.cat((tokens.input_ids, tokens.input_ids.new_tensor([result])), dim=1), stats
    finally:
        for handle in handles:
            handle.remove()
