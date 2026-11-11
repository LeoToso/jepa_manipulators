# jepa_manipulators

JEPA world models for Franka FR3 manipulation data, and tools to inspect what their latent
spaces encode. The question behind the code: when one JEPA is trained across semantically
different task families (pick-and-place vs. rotate), which task-relevant structure does the
latent keep, does it favour one family's physics over the other's, and how should the
latent be organized to transfer to new or composed tasks?

## 1. The dataset

`dataset_franka_arm.h5` merges all recordings: 300 episodes, 50 per task.

```
@tasks = [pick_and_place_banana, pick_and_place_chips, pick_and_place_orange,
          rotate_banana, rotate_chips, rotate_orange]
@episodes_per_task = 50
episode_0000000 ... episode_0000299   (task = index // 50 unless the episode has a task attribute)
```

Each episode, recorded at about 10 Hz, with all streams sharing the same timestamps:

| Key | Shape | Meaning |
|---|---|---|
| `cams/<serial>/rgb` | (T, 376, 672, 3) uint8 | camera images (gzip-compressed) |
| `fr3/q`, `fr3/dq` | (T, 7) | joint angles [rad], joint velocities [rad/s] |
| `fr3/x_ee` | (T, 7) | end-effector position [m] + quaternion (w, x, y, z) |
| `fr3/v_ee` | (T, 6) | end-effector velocity: linear [m/s] + angular [rad/s], base frame |
| `fr3/gripper` | (T,) | gripper state: -1 open, +1 closed; ramps over about 1–1.5 s while moving |
| `action/x_ee_des` | (T, 7) | target end-effector pose |
| `action/gripper` | (T,) | gripper command: -1 open, +1 close (always exactly ±1) |
| `timestamps` | (T,) | seconds since the start of the episode |

## 2. Models

All six models share the encoder, predictor, data and optimizer (`configs/base.yaml`).
They differ only in the objective:

| Config | Prediction | Second term |
|---|---|---|
| `1sp_sigreg` | one-step (teacher-forced) | SIGReg |
| `msp_sigreg` | multi-step (autoregressive, H = 3) | SIGReg |
| `1sp_epidm` | one-step | endpoint inverse dynamics (EP-IDM) |
| `msp_epidm` | multi-step | EP-IDM |
| `1sp_gcb` | one-step | goal-conditioned bisimulation (reward-free) |
| `msp_gcb` | multi-step | goal-conditioned bisimulation (reward-free) |
| `msp_gcb_sigreg` | multi-step | GCB + SIGReg (GCB alone collapsed to ~2 effective dimensions) |
| `msp_sigreg_sf` | multi-step | SIGReg + goal-conditioned successor features (BeJEPA) |

- **Architecture:** identical to [JEPA_control](https://github.com/LeoToso/JEPA_control).
  JEPA_control's encoder, predictor, action embedder and EP-IDM load into these modules with
  strict parameter-name matching and give exactly the same outputs.
  - Encoder: ViT-Tiny (128×128 input, patch 8, depth 4, 3 heads, dropout 0.1) on
    `[o_{t-1}, o_t, o_t − o_{t-1}]`. The CLS feature is concatenated with the normalized
    proprio vector, then passed through the 2048-wide BatchNorm projector to give a 192-d `z_t`.
  - Proprio (`proprio_keys`): 7 joint angles, 7 joint velocities and the gripper state
    (15-d), normalized with the training mean and std, as JEPA_control's `state_mean`/`state_std`.
    `--set model.use_proprio=false` trains a pixels-only encoder.
  - Predictor: the 6-layer causal AdaLN-zero transformer (`ARPredictor`).
    - 1SP is teacher-forced over the window, so the prediction of z_{t+1} attends to z_0..z_t,
      as in JEPA_control's `train.py`. `one_step_context: markov` uses z_t alone.
    - MSP and planning call the predictor one token at a time, as in JEPA_control.
- **Actions:** one latent step is 2 recorded steps (5 Hz). Its action is the 2 per-step actions
  `[dpos (3), drot (3), gripper command (1)]` concatenated into 14-d, like JEPA_control's
  PointMaze multi-action. dpos and drot are the achieved end-effector motion in the base frame.
  - Actions are divided by their per-dimension RMS without centering (JEPA_control's
    `action_scale`), so 0 means "no motion".
  - To execute a planned action, command `x_ee_des = x_ee ⊕ a`.
- **Kept from my version:** the real previous frame is used for the first step's frame
  difference (JEPA_control's `train.py` uses the frame itself, so that difference is 0), and
  BatchNorm statistics are recomputed every epoch (see below).
- **Proprio and the probes:** with proprio in the encoder, robot-state probes (joints, EE pose,
  gripper) partly measure the proprio input. Object identity (banana / chips / orange) is only
  visible in the images. Compare with a pixels-only run to separate the two.
- **SIGReg:** JEPA_control's variant with λ = 1, as in your paper (`sigreg_impl: jepa_control`, 128
  random directions per step). `sigreg_impl: lejepa` selects the official LeJEPA form (Epps–Pulley
  statistic, trapezoid rule on [0, 3], Gaussian weight, multiplied by N). At the same λ the official
  form is about 300–750× stronger; LeJEPA recommends λ = 0.05 for it. With proprio on the real data,
  official λ = 1 gave a useless predictor (relative rollout error about 0.97, flat for 30 epochs),
  and its latents were Gaussian only on training batches (SIGReg 0.6 on training vs. 77 on validation).
- **EP-IDM:** predicts the H action chunks from `(z_t, z_{t+H})` only.
- **GCB** ([Hansen-Estruch et al. 2022](https://arxiv.org/abs/2204.13060)):
  - `ψ(s) = z` is the JEPA latent. As in their Fig. 3, `φ(s, g)` has its own image encoder (same
    ViT) and an MLP on `[f_s, f_g, f_g − f_s]`; φ never depends on ψ.
  - Metric loss (their Eq. 6): `(d(φ(s_i,g_i), φ(s_j,g_j)) − (1−γ)|r_i − r_j| − γ d(φ̄(s'_i,g_i), φ̄(s'_j,g_j)))²`,
    with j a random permutation of the batch and the recorded next state s'.
    The (1−γ) factor comes from their Prop. 6.1. Without it, since consecutive frames have
    nearly equal φ, the fixed point is about |Δr|/(1−γ) = 100× the reward scale. In tests that
    slowly inflated the metric, and the analogy loss inflated the latent with it.
  - Analogy loss (Eq. 7): `‖(φ̄(s,g) − φ̄(g,g)) − (z_g − z_s)‖²`. This is the only GCB term that
    shapes the JEPA latent. The reward decoder (App. D.5) is also included.
  - **Hindsight goals:** g is a later frame of the same episode (offset k ~ Geometric(0.3), up
    to 30 latent steps), and s, s' and g go through each encoder in one pass.
  - **Reward-free reward** (`gcb_reward`, no task reward is used anywhere):
    - `state_distance` (default): r = −(EE position / 5 cm + EE rotation angle / 0.5 rad +
      gripper difference / 2) / 3, measured between s' and g.
    - `latent_distance`: r = −mean|z̄' − z̄_g|, purely self-supervised.
    - `indicator`: the paper's r = 1[s' = g]. With hindsight goals on logged trajectories its
      bisimulation fixed point depends only on the steps-to-goal k:
      `d = (γ^(k_i−1) + γ^(k_j−1))·1[k_i ≠ k_j]`. It therefore cannot separate tasks, objects or
      motions, and in tests the metric stayed at its trivial constant solution.
  - **Distance:** d is the per-dimension mean |·| on both sides of Eq. 6 (`gcb_distance: normalized`).
    - Using the same norm on both sides makes the update a γ-contraction (their Prop. 6.1).
      Mixing mean |·| with RMS is not one, since mean|x| ≤ RMS(x), and it diverged.
    - Per-dimension averaging makes the metric's unit match a unit-scale latent. With the
      paper's raw sums (`gcb_distance: paper`), the reward fixes distances of order 1 summed over
      all 192 dimensions; the joint optimum with the prediction MSE is then to shrink z until
      prediction is free. In tests the latent std fell to 0.004.
  - **Defaults: γ = 0.9, reward scale 10.** Without SIGReg, only the GCB anchor holds the latent's
    scale against the prediction MSE, which always prefers a smaller latent. In 10-epoch tests:
    - γ = 0.99 (paper): the metric bootstraps too slowly and the latent shrank (std 0.02,
      relative rollout error 0.59).
    - γ = 0.9, reward scale 1: latent std 0.036, relative rollout error 0.40.
    - γ = 0.9, reward scale 10: latent std 0.17 and growing, relative rollout error 0.30 and falling.

    GCB is the experimental arm. Watch `latent_std` and `rel_err_multi_step` in the training log,
    and tune `gcb_gamma`, `gcb_reward_scale` and `lambda_gcb_psi` with `--set`.
- **SF** (successor features, after BeJEPA, ICLR 2027 submission):
  - A successor predictor S(u_t, u_g) ≈ (1−γ) Σ_k γ^k u_{t+k}, where u = √d · z/‖z‖ is the
    sphere-projected JEPA latent.
  - Trained by TD toward `sg[(1−γ) u_t + γ S̄(u_{t+1}, u_g)]`, with an EMA target network
    (τ = 0.005) and γ = 0.95.
  - Hindsight goals as for GCB; the goal is absorbing, S(g, g) = u_g. Gradients reach the encoder
    through the prediction's inputs.
  - Reward-free by construction: r_g(s) = ⟨u, u_g⟩ and the goal value is V_g = ⟨S(u, u_g), u_g⟩.
  - Evaluation adds value monotonicity along episodes, and CEM toward far goals with the value
    objective vs. latent distance.
- **BatchNorm:** running statistics are recomputed at the end of every epoch ("precise BN").
  Without this, the eval-mode encoder drifts far from the train-mode one (the pre-BN mean
  is invisible to every loss), and validation, probes and planning break.

## 3. Commands

```bash
pip install -r requirements.txt

# 1) Build the cache once (all frames resized to 128x128, about 6 GB for 120k frames).
#    The train/val/test split (40/5/5 episodes per task) is fixed here and shared by all models.
python -m jepa.data /path/to/dataset_franka_arm.h5 --out /mnt/t7shield/cache_128 --image-size 128

# 2) Train (one command per model)
python -m jepa.train --cache /mnt/t7shield/cache_128 --config configs/1sp_sigreg.yaml --save-dir runs/1sp_sigreg
python -m jepa.train --cache /mnt/t7shield/cache_128 --config configs/msp_sigreg.yaml --save-dir runs/msp_sigreg
python -m jepa.train --cache /mnt/t7shield/cache_128 --config configs/1sp_epidm.yaml  --save-dir runs/1sp_epidm
python -m jepa.train --cache /mnt/t7shield/cache_128 --config configs/msp_epidm.yaml  --save-dir runs/msp_epidm
python -m jepa.train --cache /mnt/t7shield/cache_128 --config configs/1sp_gcb.yaml    --save-dir runs/1sp_gcb
python -m jepa.train --cache /mnt/t7shield/cache_128 --config configs/msp_gcb.yaml    --save-dir runs/msp_gcb

# 3) Evaluate each model, then compare them
for m in 1sp_sigreg msp_sigreg 1sp_epidm msp_epidm 1sp_gcb msp_gcb; do
  python -m jepa.evaluate --ckpt runs/$m/model_final.pt --cache /mnt/t7shield/cache_128 --out eval/$m
done
python -m jepa.compare --evals eval/* --runs runs/* --out eval/compare
```

Useful training options:
- **`--amp`:** bfloat16 autocast, much faster on A100/H100/RTX 30xx or newer GPUs.
- **`--resume`:** continue from `runs/<model>/checkpoints/last.pt`.
- **`--set section.key=value`:** override any config entry, e.g. `--set training.batch_size=128 data.horizon=5`.
- **Zero-shot hold-out:** train on a subset of tasks with
  `--set "data.tasks=[pick_and_place_banana, pick_and_place_chips, rotate_banana, rotate_chips]"`,
  then evaluate on all tasks.

`runs/<model>/` contains `model_final.pt` (use this for comparisons), `model_best.pt` (lowest
validation loss of that model's own objective), `history.json` (losses plus a quick probe every
5 epochs) and `checkpoints/epochXXXX.pt`.

## 4. What the evaluation measures

`jepa.evaluate` works on the held-out test episodes and fits its probes on training episodes.
Figures go to `eval/<model>/figs/`; every number is in `eval/<model>/metrics.json`.

| | Question | Analysis |
|---|---|---|
| A | What is in `z_t`? | Linear and MLP probes: EE position / rotation / velocities, gripper, joints, task progress. Linear and kNN accuracy for task, family and object. Action R² from `(z_t, z_{t+1})` |
| B | Is one family's physics favoured? | The same probes evaluated, and refitted, inside each family: position on pick-and-place vs. rotation and angular velocity on rotate |
| B | What survives prediction? | Probes applied to k-step predictions `ẑ_{t+k}` (k = 1..10), rollout error vs. k per family, task accuracy from predictions |
| C | How is the space organized? | Covariance spectrum and effective rank (overall, per task), 1-D projections vs. N(0, 1) and Epps–Pulley statistic (LeJEPA), PCA / t-SNE coloured by task, progress, gripper and height, overlap of per-task principal subspaces, between/within-task variance, temporal straightening and transition r95 |
| D | Is the geometry goal-aware? | Monotonicity of `‖z_t − z_goal‖` along episodes (planning-cost shape), a steps-to-goal probe from `(z_t, z_g)`, GCB-style analogies `z(B_0) + z(A_T) − z(A_0)` → nearest neighbour (same task / same family with another object) |
| E | Is it usable for planning? | Offline CEM towards the frame 5 latent steps ahead, compared with the recorded actions; probability that random action sequences score lower than the recorded one |
| F | What does the encoder look at? | First 3 principal components of the patch tokens as RGB (LeJEPA Fig. 14) |

`jepa.compare` puts all models on the same axes (`cmp_*.png`), writes `summary.md` / `summary.csv`,
and plots training loss vs. probe accuracy over epochs (LeJEPA Fig. 1).

Not covered yet:
- **Object orientation:** the dataset has no object pose labels, so rotation information is
  measured through the end-effector pose, which follows the object while it is grasped.
- **Closed-loop success rates:** these need the robot (or a simulator). The planner in
  `jepa/evaluate.py` (`cem_plan`) is the piece to reuse for deployment.

## 5. Data tools (earlier dataset)

```bash
python inspect_dataset.py Dataset                         # what is in a dataset (any format)
python view_episode.py Dataset/<file>.h5 --episode all --png
python extract_frames.py Dataset/<file>.h5 --episode 0 --num 10
python franka_dataset.py check-quat Dataset               # quaternion order (wxyz on this robot)
```
