"""JEPA world models for the multi-task Franka FR3 dataset.

data      cache building and windowed sampling (actions, hindsight goals)
models    encoder / predictor / inverse model / GCB heads (adapted from JEPA_control)
losses    one-step & multi-step prediction, SIGReg, EP-IDM, goal-conditioned bisimulation
train     training entry point          (python -m jepa.train ...)
evaluate  latent-space analysis suite   (python -m jepa.evaluate ...)
compare   cross-model figures/tables    (python -m jepa.compare ...)
"""
