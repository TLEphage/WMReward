# MAGI-1-4.5B WMReward Comparison 0001-0008

- Lower surprise is better; higher similarity is better.
- Baseline surprise mean: 0.430063
- Once-guidance surprise mean: 0.438277 (+0.008214 vs baseline, 2/8 improved)
- gf=5 guidance surprise mean: 0.429572 (-0.000491 vs baseline, 5/8 improved)
- gf=5 vs once-guidance surprise mean delta: -0.008705 (6/8 lower than once guidance)

| video | baseline | once | gf=5 | once-base | gf5-base | gf5-once |
|---|---:|---:|---:|---:|---:|---:|
| 0001_trimmed-ball-and-block-fall.mp4 | 0.454181 | 0.458469 | 0.454822 | +0.004288 | +0.000641 | -0.003647 |
| 0002_trimmed-ball-and-block-fall.mp4 | 0.439220 | 0.432365 | 0.439641 | -0.006855 | +0.000421 | +0.007276 |
| 0003_trimmed-ball-and-block-fall.mp4 | 0.432139 | 0.447834 | 0.431881 | +0.015695 | -0.000258 | -0.015953 |
| 0004_trimmed-ball-behind-rotating-paper.mp4 | 0.441799 | 0.435288 | 0.442057 | -0.006511 | +0.000258 | +0.006769 |
| 0005_trimmed-ball-behind-rotating-paper.mp4 | 0.444127 | 0.457647 | 0.443585 | +0.013520 | -0.000542 | -0.014062 |
| 0006_trimmed-ball-behind-rotating-paper.mp4 | 0.423192 | 0.440295 | 0.420423 | +0.017103 | -0.002769 | -0.019872 |
| 0007_trimmed-ball-hits-duck.mp4 | 0.416143 | 0.439664 | 0.416057 | +0.023521 | -0.000086 | -0.023607 |
| 0008_trimmed-ball-hits-duck.mp4 | 0.389707 | 0.394655 | 0.388112 | +0.004948 | -0.001595 | -0.006543 |
