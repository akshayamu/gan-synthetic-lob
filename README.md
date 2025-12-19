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


Distributional Evaluation (Real vs Synthetic)

We evaluate synthetic LOB realism using distributional divergence metrics on key
one-dimensional marginals commonly used in recent limit order book generation
literature. Specifically, we report the Kolmogorov–Smirnov (KS) statistic and
Wasserstein-1 distance (W1) between real and synthetic distributions.

Evaluated Marginals

Bid–Ask Spread

Total Book Depth (sum of bid and ask volumes)

(Mid-price returns evaluated where applicable)
Quantitative Results
Marginal	KS Statistic	Wasserstein-1
Spread	              0.58	  0.026
Depth	              0.49	  3.02

Lower values indicate closer alignment between real and synthetic distributions.

Interpretation

While vanilla GAN models exhibited poor alignment on these marginals, replacing the
objective with a Wasserstein GAN with Gradient Penalty (WGAN-GP) and restricting the
learning target to economically meaningful quantities significantly reduced
distributional divergence. These results are consistent with prior findings that
stable objectives and reduced targets are critical for synthetic LOB generation under
limited data regimes.

Quantitative Evaluation (Real vs Synthetic LOB)

To assess realism beyond visual inspection, we compare key microstructure features
between real and GAN-generated limit order book (LOB) snapshots.

Features evaluated (per snapshot):

Bid–ask spread

Total bid depth (sum of top-10 bid sizes)

Total ask depth (sum of top-10 ask sizes)

We report summary statistics and the Kolmogorov–Smirnov (KS) statistic
to quantify distributional similarity
| Feature         | Real Mean | Synthetic Mean | KS Statistic |
| --------------- | --------- | -------------- | ------------ |
| Spread          | 1.33      | 0.57           | **0.77**     |
| Total Bid Depth | 1.60e-4   | 1.64e-4        | **0.58**     |
| Total Ask Depth | 1.69      | 1.73           | **0.52**     |

Notes:

Depth statistics show close alignment in first-order moments.

Spread distribution exhibits mode collapse, reflected by a high KS statistic.

These results motivate future improvements such as WGAN-GP objectives and
temporal conditioning.

This evaluation ensures the model’s limitations are quantified and transparent.