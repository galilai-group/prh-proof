# Design: what comes from stable-pretraining, and why the rest exists

The training lifecycle is stable-pretraining's (every run of the paper used revision `ab836bf699a2be712dfcf980a5eb70d36391e876`):

| piece | stable-pretraining component |
|---|---|
| JEPA objective (invariance to the global views' mean + SIGReg), projector, ViT backbones | `stable_pretraining.methods.lejepa.LeJEPA` |
| training and validation loops, optimizer, warmup-cosine schedule, seeding, run directories, `last.ckpt`, SLURM requeue, DDP | `spt.Module`, `spt.Manager` |
| multi-crop views, photometric augmentation on GPU | `spt.data.transforms.MultiViewTransform`, `RandomResizedCrop`, `spt.data.gpu_transforms` |
| datasets and loaders | `spt.data.FromTorchDataset`, `spt.data.HFDataset`, `spt.data.DataModule` |
| validation embedding dumps, rank monitoring | `spt.callbacks.OnlineWriter`, `spt.callbacks.RankMe` |
| random-initialized BERT, frozen encoders | `spt.backbone.utils.from_huggingface`, `EvalOnly` |

Every arm of the paper is the same `train.py` call with different arguments: the backbone (ViT or BERT of the
same width), the views, and the objective. Each piece of custom code below was added only after checking that
stable-pretraining does not provide it.

| custom code | what it does | evidence that stable-pretraining lacks it |
|---|---|---|
| `prh.views.TextView` | span crop and token masking of a caption; draws one caption from a list (good views) | every masking transform in `data/transforms.py`, `data/masking.py`, `data/collate.py` (`RandomMask`, `PatchMasking`, `ContextTargetsMultiBlockMask`, `multi_block_mask`) needs an H×W image; nothing under `stable_pretraining/data` uses a tokenizer |
| `prh.views._pick` | an image view of a clip draws its own frame (good image views) | `UniformTemporalSubsample` is deterministic and `RandomContiguousTemporalSampler` returns contiguous frames of one tensor; neither draws one frame per view |
| `prh.data.extract_frames`, `MSRVTTClips` | 8 frames per MSR-VTT clip as JPEGs, 20 captions per clip | `build_lance_video_dataset` stores every frame (about 4.5M for MSR-VTT, ~45 GB) and `LanceVideoSegments` uses one fixed frame stride for clips of different lengths |
| `prh.data.CocoCaptions`, `RecapText`, `RecapImages` | text-only caption datasets; Recap tar shards read in place | no COCO-caption or Recap-DataComp reader |
| `prh.data.EpochSampler` | a random subset of fixed size per epoch, on any number of GPUs | `data/sampler.py` has `RepeatedRandomSampler` (a permutation of the whole dataset), `SupervisedBatchSampler` and `RandomBatchSampler` (batch samplers over the whole dataset); torch's `RandomSampler(num_samples=...)` works on one GPU but Lightning replaces it by a `DistributedSampler` of the whole dataset under DDP, so it delegates to `RandomSampler` without being one |
| `prh.forward.next_token` | next-token prediction with a tied softmax head on the same BERT | `methods/igpt.py` and `methods/nepa.py` are image-patch only; no text language-model loss under `losses/` |
| `prh.models.CaptionDecoder`, `LMDecoder`, `prh.forward.decode` | text-only caption decoders on a frozen text JEPA: a small transformer from scratch, or a pretrained language model with a soft prompt; beam search | no captioning component; `from_huggingface` builds encoders, not prefix-conditioned generators |
| `prh.views._rgb` | PIL RGB conversion before crops | `ToImage(rgb=True)` leaves CMYK ImageNet JPEGs with 4 channels |
| `prh.views.Photometric` | stable-pretraining's GPU colour ops, applied once per resolution on the stacked views | kornia's GPU crop samples its parameters on the CPU (about 130 ms per view), so crops stay on the CPU workers and only the colour operations run on the GPU, one call per resolution on the stacked views (0.76 to 0.29 s per 8-view ViT-S step) |
| `prh.views.image_eval_view` | resize 256, centre crop 224, ImageNet normalization | a composition of stable-pretraining transforms, shared by training validation and frozen-encoder embedding |
| `prh.features` reload of an image JEPA | rebuilds `LeJEPA` from a checkpoint to embed new images | `Manager` resumes training runs; it does not expose a frozen backbone for inference outside a run |
| `prh.align`, `prh.metrics`, `prh.features` | rotations between spaces, mutual k-NN, retrieval, the PRH-paper protocol | `grep -ri "procrustes\|mutual_knn\|cka" stable_pretraining/` finds nothing |
| `prh.theory` | the paper's finite-world statements | mathematics, not training code |
| `prh.data.overlap`, `prh.data.paraphrase` | evaluation pairs among the training data; text-only paraphrases of alt-texts | data preparation outside training; no deduplication or text-generation utility under `stable_pretraining/data` |

Two settings were fixed for stability on long text schedules: gradient clipping needs the non-fused AdamW
(`--clip` sets `fused=False`), and some machines cannot build cuDNN attention plans
(`torch.backends.cuda.enable_cudnn_sdp(False)`).
