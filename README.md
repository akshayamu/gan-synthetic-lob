# Synthetic LOB GAN Project

## Experiment Progression

This project was developed iteratively, with each model choice motivated by
quantitative evaluation rather than visual inspection.

1. **Vanilla GAN (Full LOB, 40 features)**  
   Initial experiments trained a standard GAN on full multi-level LOB snapshots.
   Quantitative tests revealed poor distributional alignment on economically
   meaningful quantities (high KS statistics on spread and depth).

2. **Improved GAN (Stabilization Techniques)**  
   Architectural changes (deeper networks, dropout, label smoothing) improved
   training stability but did not materially improve realism or downstream
   performance.

3. **WGAN-GP on Reduced Targets (Final Model)**  
   Guided by evaluation results, the learning target was reduced to key LOB
   marginals (spread and total depth), and the loss was replaced with a
   Wasserstein objective with gradient penalty (WGAN-GP). This resulted in
   substantial reductions in distributional divergence.

