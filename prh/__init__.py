"""A simple proof of the Platonic Representation Hypothesis for JEPAs: code.

Training (loops, optimization, checkpoints, SLURM requeue, probes) is stable-pretraining's; ``train.py`` wires it.
This package holds only what stable-pretraining does not provide:

* :mod:`prh.theory`: the paper's finite-world statements, executable;
* :mod:`prh.views`: good and bad text views (image views are stable-pretraining multi-crop);
* :mod:`prh.models`: the text encoder and the text-only caption decoders;
* :mod:`prh.forward`: forward functions of the three objectives (JEPA, next-token, caption decoder);
* :mod:`prh.data`: training datasets and evaluation tables;
* :mod:`prh.features`: embeddings from training dumps or frozen encoders;
* :mod:`prh.align`: the rotation between two embedding spaces;
* :mod:`prh.metrics`: alignment, captioning and text-similarity scores.
"""

from . import align, data, features, forward, metrics, models, theory, views
from .align import Rotation, procrustes
from .features import embed_images, embed_texts, last_embeddings, load_embeddings, save_embeddings
from .metrics import alignment, caption_scores, matched_noise, mutual_knn, prh_score, recall
from .models import ARCH, TOKENIZER, CaptionDecoder, LMDecoder, TextEncoder, load_decoder
from .views import TextView, view_transforms

__all__ = [
    # modules
    "align",
    "data",
    "features",
    "forward",
    "metrics",
    "models",
    "theory",
    "views",
    # models and views
    "ARCH",
    "TOKENIZER",
    "CaptionDecoder",
    "LMDecoder",
    "load_decoder",
    "TextEncoder",
    "TextView",
    "view_transforms",
    # embeddings
    "embed_images",
    "embed_texts",
    "last_embeddings",
    "load_embeddings",
    "save_embeddings",
    # alignment
    "Rotation",
    "procrustes",
    "alignment",
    "mutual_knn",
    "prh_score",
    "recall",
    # captioning
    "caption_scores",
    "matched_noise",
]
