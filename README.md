# From Groups to Rings: Reproducibility Code

This minimal, self-contained package accompanies *From Groups to Rings: Causal Evidence for Algebraic Decomposition in Grokked Transformers*. It provides runnable code for representative finite group-algebra multiplication experiments and post-attention component-subspace interventions. It does not depend on the authors' internal project structure and contains no checkpoints or precomputed results.

## Environment

Python 3.9 or newer is recommended. The release was validated with Python 3.9.6, NumPy 2.0.2, scikit-learn 1.6.1, and PyTorch 2.7.1; `requirements.txt` gives compatible lower bounds rather than forcing a platform-specific PyTorch build.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts select CUDA, MPS, or CPU automatically, in that order. A device may also be selected explicitly with `--device cuda` or `--device cpu`.

## Quick smoke test

The following command checks data generation, the model forward/backward pass, and checkpoint writing. Two steps are not intended to produce grokking.

```bash
python3 train.py --algebra F3_Z4 --steps 2 --eval-interval 1
```

## Representative training runs

```bash
python3 train.py --algebra F2_Z7 --seed 42
python3 train.py --algebra F3_Z4 --seed 42
python3 train.py --algebra F2_Z6 --seed 42
```

The default model is a one-layer Transformer without LayerNorm, with `d_model=128`, 4 attention heads, and `d_mlp=512`. Training uses AdamW with learning rate `1e-3`, weight decay `1.0`, a 30% training split, and at most 100,000 optimization steps. All defaults are recorded in `config.py`, and each run writes its effective configuration to `outputs/<algebra>_seed<seed>/config.json`.

## Post-attention component-subspace intervention

After a training run has completed, use:

```bash
python3 analyze_intermediate.py \
  --run-dir outputs/F3_Z4_seed42 \
  --n-interventions 2000 \
  --n-random 20
```

The analysis:

1. exhaustively verifies joint injectivity and multiplicative homomorphism of the listed CRT projections;
2. collects answer-token activations immediately after attention;
3. holds out complete product elements using a component-stratified split;
4. identifies a fixed low-dimensional subspace from logistic-probe weights using SVD;
5. evaluates the fixed projector on held-out matched interventions; and
6. compares it with 20 same-dimensional random subspaces.

The fixed seeds, sampling procedure, metric definitions, and random-control summaries reproduce the protocol used for the author-response table. The script reports component target accuracy, exact counterfactual accuracy, and non-target preservation, together with their conditional counterparts. It saves the complete metrics to `intermediate_subspace_metrics.json` in the selected run directory and prints a grep-friendly numerical summary to standard output.

The author-response runs used the MPS backend. Training trajectories and probe-defined subspaces can show backend-level floating-point variation, particularly for the weaker, method-sensitive components; the training backend is recorded in each run's `config.json`, and the analysis output records its own device.

The probe/SVD sensitivity check can be reproduced by changing `--variance-threshold` among `0.80`, `0.90`, `0.95`, and `0.99`, and changing `--probe-kind` between `logistic` and `ridge`. Non-default settings are written to separately tagged JSON files so that results are not overwritten.

These intervention results are intended as a causal diagnostic. Mixed component-level results should not be interpreted as complete or probe-independent identification of a direct-sum decomposition.

## Files

- `algebra.py`: multiplication tables, CRT projections, and exhaustive projection audits;
- `model.py`: Transformer implementation with intermediate-state patching;
- `train.py`: training, evaluation, and checkpoint saving;
- `analyze_intermediate.py`: cross-fitted component-subspace interventions and random controls;
- `config.py`: supported algebras and default hyperparameters;
- `requirements.txt`: Python dependencies.
