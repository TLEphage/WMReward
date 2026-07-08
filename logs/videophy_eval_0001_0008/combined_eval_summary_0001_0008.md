# WMReward MAGI-1-4.5B Evaluation Summary

## VideoPhy / VideoCon-Physics, videos 0001-0008
| method | n | SA mean | PC mean | avg mean | SA>=0.5 | PC>=0.5 | both>=0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 8 | 0.202377 | 0.064667 | 0.133522 | 0.000000 | 0.000000 | 0.000000 |
| once_guidance | 8 | 0.243481 | 0.099640 | 0.171560 | 0.125000 | 0.000000 | 0.000000 |
| guidance_gf5 | 8 | 0.240143 | 0.071350 | 0.155746 | 0.000000 | 0.000000 | 0.000000 |

VideoPhy official inference outputs continuous entailment probabilities. The pass-rate columns above use score >= 0.5 only as a binary proxy.

## Physics-IQ official evaluator subset, videos 0001-0006
| method | origround / 100 | orig [0,1] | stable [0,1] | view [0,1] |
| --- | --- | --- | --- | --- |
| baseline | 30.490000 | 0.304860 | 0.304860 | 0.315570 |
| once_guidance | 25.700000 | 0.256943 | 0.256943 | 0.304078 |
| guidance_gf5 | 30.450000 | 0.304524 | 0.304524 | 0.315237 |

Physics-IQ official scoring is reported only for 0001-0006 because it needs complete 3-view scenarios. Current 0007-0008 are two views of the next scenario and lack 0009/right view. Generated clips used for this official evaluator were padded from 3s/24 frames to 5s/40 frames.

## Files
- Per-video VideoPhy scores: `logs/videophy_eval_0001_0008/videophy_scores_0001_0008.tsv`
- VideoPhy method summary: `logs/videophy_eval_0001_0008/videophy_method_summary_0001_0008.tsv`
- Physics-IQ subset summary: `logs/videophy_eval_0001_0008/physicsiq_official_subset_0001_0006_summary.tsv`
- SA log: `logs/videophy_eval_0001_0008/run_sa.log`
- PC log: `logs/videophy_eval_0001_0008/run_pc.log`
