Goal: Build a custom neural network, train on synthetic periodic data, prove it learns.
Novel component (v1): Sine interference layer.
Why it might work: Sine creates periodic features, potentially better suited for wave/cycle data.
V1 final val loss: 0.3321
V2 final val loss (lr=1e-3): 0.3583
V2 final val loss (lr=5e-4): 0.4566
Best V2: 0.3583
Conclusion: Learnable omega did not improve over fixed-frequency baseline on this task.
Interpretation: Adding per-neuron frequencies increased parameter count and training instability without measurable benefit.
--- Phase 7: Comparison chart ---
compare.py final v1 val loss: <paste>
compare.py final v2 val loss: <paste>
Difference (v2 - v1): <paste>
Chart saved: comparison.png