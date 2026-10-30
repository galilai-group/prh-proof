# Agent operating mode

  Operate autonomously, but minimize new code and experiments.

  ## Highest-priority rule: use stable-pretraining

  This repository uses the sibling `../stable-pretraining` project as its training
  framework. These instructions override all later autonomy and implementation
  instructions.

  Never implement or duplicate functionality already provided by
  stable-pretraining.

  Stable-pretraining owns:

  - training, validation, and test loops;
  - Lightning modules and Trainer integration;
  - `stable_pretraining.Manager`;
  - optimizer and scheduler construction;
  - datasets, samplers, transforms, and multi-view batching;
  - online probes and evaluation callbacks;
  - checkpointing, resumption, logging, and run registries;
  - distributed training, DDP, SLURM lifecycle, and requeue handling;
  - standard SSL methods and losses.

  A new repo should normally contain only:

  - the novel mathematical loss, estimator, or forward computation;
  - project-specific model components unavailable in stable-pretraining;
  - thin configuration or wiring around stable-pretraining;
  - focused tests for that novel functionality.

  Do not create custom Trainers, training steps, evaluation loops, replay buffers,
  checkpoint managers, distributed lifecycle code, probe implementations, campaign
  frameworks, or job orchestration when stable-pretraining provides the capability.

  ## Mandatory discovery before implementation

  Before writing training-related code:

  1. Read `stable-pretraining/AGENTS.md`.
  2. Inspect `stable-pretraining/METHODS.md`.
  3. Search stable-pretraining's methods, forward functions, losses, callbacks,
     data utilities, examples, and configuration support.
  4. Search this repository for the existing stable-pretraining entrypoint and
     closest working experiment.
  5. Identify the exact installed or pinned stable-pretraining revision.
  6. State which stable-pretraining components will be reused.

  Do not infer that a feature is missing merely because its API is unfamiliar.

  Reuse in this order:

  1. an existing stable-pretraining method class;
  2. an existing forward function, loss, or callback;
  3. an existing example or YAML configuration with overrides;
  4. a minimal adapter around stable-pretraining;
  5. new framework code only with explicit user approval.

  If stable-pretraining almost supports the requirement, extend or configure it
  instead of creating a parallel implementation.

  ## Experiment rules

  A new experimental condition should ordinarily be a configuration change, not a
  new training script.

  All comparison arms must use the same stable-pretraining entrypoint, data
  pipeline, evaluator, optimizer policy, seed policy, and training lifecycle.
  Change only the factor being tested.

  For example, original loss versus stable loss should differ only in the selected
  loss or forward component. Supervised, DINO, YY^T, and VLAD baselines must reuse
  the same framework and evaluation path wherever applicable.

  Before submitting a full run:

  1. print and inspect the resolved configuration;
  2. run a short local or single-job smoke test;
  3. verify that training, validation, checkpointing, and metrics come from
     stable-pretraining;
  4. verify that the expected dataset, transforms, loss, evaluator, and resume
     behavior are active;
  5. submit only the experiment requested by the user.

  Do not launch additional controls, sweeps, or alternative implementations unless
  the user requested them.

  ## Custom-code gate

  Before adding custom training-related code, record concrete evidence that the
  capability is absent from stable-pretraining:

  - paths and symbols searched;
  - nearest existing stable-pretraining component;
  - why configuration or a thin adapter is insufficient;
  - the smallest proposed addition.

  If this evidence is unavailable, do not implement the custom code.

  Any unavoidable adapter must delegate lifecycle behavior to
  `stable_pretraining.Manager` and remain small enough to review directly.

  ## Debugging

  When a result regresses unexpectedly, first assume configuration or integration
  drift rather than inventing a replacement implementation.

  Compare the failing run against the last known-good run:

  - resolved configuration;
  - stable-pretraining revision;
  - entrypoint;
  - dataset and transforms;
  - optimizer and scheduler;
  - loss parameters;
  - evaluator;
  - checkpoint and resume state;
  - precision, devices, and DDP settings.

  Reproduce the known-good baseline through the existing stable-pretraining path
  before changing source code.

  Never introduce replay buffers, alternate estimators, custom lifecycle logic, or
  new training machinery merely to explain or repair a regression.

  ## Mathematical work

  For a genuinely novel mathematical component:

  1. derive the component independently of the training framework;
  2. compare it against a slow reference implementation;
  3. test adversarial and synthetic cases;
  4. test gradients and numerical behavior;
  5. integrate it as a loss, forward function, callback, or model component inside
     stable-pretraining;
  6. run a minimal end-to-end smoke test using stable-pretraining.

  Mathematical validation does not authorize reimplementing the surrounding
  training system.

  ## Subagents

  Use subagents only for bounded, non-overlapping tasks such as mathematical
  review, test falsification, or focused code review.

  Do not ask multiple agents to independently build training pipelines.
  All subagents must follow the stable-pretraining-first rules above.
  The main agent must prevent duplicate implementations and unnecessary jobs.

  ## Repository safety

  Inspect freely and run focused tests.

  Do not destroy unrelated work, rewrite repository history, cancel jobs belonging
  to other users or agents, or modify stable-pretraining itself unless explicitly
  requested.

  Prefer modifying an existing configuration or entrypoint over creating a new
  script. Report every submitted job ID and the exact command or resolved
  configuration used.


