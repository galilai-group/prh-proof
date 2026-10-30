"""Forward functions for ``stable_pretraining.Module``, one per training objective.

Each is called as ``forward(module, batch, stage)`` and returns a dict with ``loss`` in training, and in
validation the tensors that ``OnlineWriter`` dumps for offline alignment (``embedding``, ``proj``, ``pair_id``,
``is_test``). The module attributes they use are set in ``train.py``: ``key`` (``"image"`` or ``"text"``), and
``model`` (LeJEPA), ``backbone`` (next-token) or ``decoder``/``encoder``/``noise`` (caption decoder).
"""

import torch
import torch.nn.functional as F


def _eval_outputs(batch, emb, proj):
    return {"embedding": emb, "proj": proj, "pair_id": batch["pair_id"], "is_test": batch["is_test"]}


def lejepa(self, batch, stage):
    """LeJEPA, modality-agnostic: invariance of every view to the global views' mean, plus SIGReg.

    The first ``self.n_global`` views are the global ones. The loss is stable-pretraining's ``LeJEPA``.
    """
    if stage == "fit":
        views = [v[self.key] for v in batch["views"]]
        out = self.model(global_views=views[: self.n_global], local_views=views[self.n_global :])
        self.log_dict({"fit/loss": out.loss, "fit/inv": out.inv_loss, "fit/sigreg": out.sigreg_loss})
        return {"loss": out.loss, "embedding": out.embedding}
    emb = self.model(images=batch[self.key]).embedding
    return _eval_outputs(batch, emb, self.model.projector(emb))


def next_token(self, batch, stage):
    r"""Next-token prediction through a softmax head tied to the input embeddings.

    Per position, :math:`-\langle h, e_y\rangle + \log\sum_v e^{\langle h, e_v\rangle}`: a JEPA prediction term
    plus a log-partition anti-collapse term (paper, Prop. 2). The embedding is the mean-pooled last hidden state.
    """
    tokens = batch[self.key]
    h = self.backbone.hidden(tokens)
    logits = h[:, :-1] @ self.backbone.bert.embeddings.word_embeddings.weight.T
    loss = F.cross_entropy(logits.flatten(0, 1).float(), tokens[:, 1:].flatten(), ignore_index=self.backbone.pad_id)
    self.log(f"{stage}/loss", loss)
    emb = self.backbone.pool(h, tokens)
    return {"loss": loss, "embedding": emb} if stage == "fit" else {"loss": loss, **_eval_outputs(batch, emb, emb)}


def decode(self, batch, stage):
    """Caption decoder on a frozen text JEPA: rebuild each caption from its own (noised) embedding.

    Training noise of relative scale ``self.noise`` makes the decoder tolerate the residual of the cross-modal
    rotation it will be applied through (as in CapDec, Nukrai et al. 2022).
    """
    tokens = batch[self.key]
    with torch.no_grad():
        z = self.encoder(tokens).float()
    if stage == "fit":
        self.decoder.update_mean(z)
    c = self.decoder.condition(z, self.noise if stage == "fit" else 0.0)
    loss = self.decoder.loss(c, tokens, batch.get("caption"))  # validation batches carry token ids only
    self.log(f"{stage}/loss", loss)
    return {"loss": loss, "embedding": c} if stage == "fit" else {"loss": loss, **_eval_outputs(batch, c, c)}
