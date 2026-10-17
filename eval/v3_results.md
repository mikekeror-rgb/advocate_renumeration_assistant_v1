# v3 results (pre-registered analysis)
236 questions; cluster bootstrap by case, 10000 resamples, seed 20261017.
Questions / clusters: ALL 236/181, ruling_unseen 69/14, ruling_seen 48/48, statute 23/23, noanswer 56/56, calc 40/40

## Accuracy by system (%, with 95% interval)
| system | ALL | ruling_unseen | ruling_seen | statute | noanswer | calc |
|---|---|---|---|---|---|---|
| base | 87.7 [83.3, 92.1] | 78.3 [68.9, 87.9] | 79.2 [66.7, 89.6] | 91.3 [78.3, 100.0] | 96.4 [91.1, 100.0] | 100.0 [100.0, 100.0] |
| base_rerun | 87.7 [83.1, 92.2] | 76.8 [66.2, 87.1] | 81.2 [68.8, 91.7] | 91.3 [78.3, 100.0] | 96.4 [91.1, 100.0] | 100.0 [100.0, 100.0] |
| ep1 | 85.2 [79.7, 90.5] | 75.4 [62.5, 87.7] | 72.9 [60.4, 85.4] | 87.0 [73.9, 100.0] | 96.4 [91.1, 100.0] | 100.0 [100.0, 100.0] |
| dpo | 86.0 [80.2, 91.6] | 72.5 [56.5, 86.4] | 81.2 [68.8, 91.7] | 87.0 [69.6, 100.0] | 96.4 [91.1, 100.0] | 100.0 [100.0, 100.0] |
| nocase | 83.1 [77.6, 88.5] | 68.1 [55.0, 80.6] | 70.8 [58.3, 83.3] | 91.3 [78.3, 100.0] | 96.4 [91.1, 100.0] | 100.0 [100.0, 100.0] |
| snip1000 | 75.8 [69.4, 82.7] | 52.2 [40.0, 64.6] | 60.4 [45.8, 75.0] | 87.0 [73.9, 100.0] | 96.4 [91.1, 100.0] | 100.0 [100.0, 100.0] |
| nocalc | 77.1 [71.4, 82.5] | 79.7 [69.1, 90.0] | 85.4 [75.0, 93.8] | 95.7 [87.0, 100.0] | 100.0 [100.0, 100.0] | 20.0 [7.5, 32.5] |
| stripped | 59.7 [53.4, 66.7] | 42.0 [32.9, 52.5] | 54.2 [39.6, 68.8] | 91.3 [78.3, 100.0] | 100.0 [100.0, 100.0] | 22.5 [10.0, 35.0] |

## H1: each pipeline component (ablation on base, full pipeline = base)

**case scoping off** (nocase minus base, points)
| subset | diff | 95% interval | b better / a better |
|---|---|---|---|
| ALL | -4.7 | [-8.8, -0.8] | 4 / 15 |
| ruling_unseen | -10.1 | [-22.1, +0.0] | 2 / 9 |
| ruling_seen | -8.3 | [-20.8, +2.1] | 2 / 6 |
| statute | +0.0 | [+0.0, +0.0] | 0 / 0 |
| noanswer | +0.0 | [+0.0, +0.0] | 0 / 0 |
| calc | +0.0 | [+0.0, +0.0] | 0 / 0 |

**context snippets cut to 1000 chars** (snip1000 minus base, points)
| subset | diff | 95% interval | b better / a better |
|---|---|---|---|
| ALL | -11.9 | [-16.9, -6.8] | 4 / 32 |
| ruling_unseen | -26.1 | [-36.9, -16.2] | 1 / 19 |
| ruling_seen | -18.8 | [-33.3, -6.2] | 2 / 11 |
| statute | -4.3 | [-17.4, +8.7] | 1 / 2 |
| noanswer | +0.0 | [+0.0, +0.0] | 0 / 0 |
| calc | +0.0 | [+0.0, +0.0] | 0 / 0 |

**calculator off** (nocalc minus base, points)
| subset | diff | 95% interval | b better / a better |
|---|---|---|---|
| ALL | -10.6 | [-16.2, -5.6] | 8 / 33 |
| ruling_unseen | +1.4 | [-3.1, +6.7] | 2 / 1 |
| ruling_seen | +6.2 | [+0.0, +14.6] | 3 / 0 |
| statute | +4.3 | [+0.0, +13.0] | 1 / 0 |
| noanswer | +3.6 | [+0.0, +8.9] | 2 / 0 |
| calc | -80.0 | [-92.5, -67.5] | 0 / 32 |

**all three removed** (stripped minus base, points)
| subset | diff | 95% interval | b better / a better |
|---|---|---|---|
| ALL | -28.0 | [-34.7, -21.2] | 8 / 74 |
| ruling_unseen | -36.2 | [-47.8, -24.3] | 0 / 25 |
| ruling_seen | -25.0 | [-41.7, -8.3] | 4 / 16 |
| statute | +0.0 | [-17.4, +17.4] | 2 / 2 |
| noanswer | +3.6 | [+0.0, +8.9] | 2 / 0 |
| calc | -77.5 | [-90.0, -65.0] | 0 / 31 |

## Noise: identical configuration rerun

**base rerun** (base_rerun minus base, points)
| subset | diff | 95% interval | b better / a better |
|---|---|---|---|
| ALL | +0.0 | [-1.2, +1.3] | 1 / 1 |
| ruling_unseen | -1.4 | [-4.5, +0.0] | 0 / 1 |
| ruling_seen | +2.1 | [+0.0, +6.2] | 1 / 0 |
| statute | +0.0 | [+0.0, +0.0] | 0 / 0 |
| noanswer | +0.0 | [+0.0, +0.0] | 0 / 0 |
| calc | +0.0 | [+0.0, +0.0] | 0 / 0 |

## H2: SFT (ep1) vs base

**SFT effect** (ep1 minus base, points)
| subset | diff | 95% interval | b better / a better |
|---|---|---|---|
| ruling_unseen | -2.9 | [-10.1, +4.9] | 4 / 6 |
| ALL | -2.5 | [-5.9, +0.9] | 7 / 13 |
| ruling_seen | -6.2 | [-16.7, +4.2] | 2 / 5 |
| statute | -4.3 | [-21.7, +8.7] | 1 / 2 |
| noanswer | +0.0 | [+0.0, +0.0] | 0 / 0 |
| calc | +0.0 | [+0.0, +0.0] | 0 / 0 |

## H3: DPO vs SFT (ep1)

**DPO effect** (dpo minus ep1, points)
| subset | diff | 95% interval | b better / a better |
|---|---|---|---|
| ruling_unseen | -2.9 | [-12.5, +5.6] | 3 / 5 |
| ALL | +0.8 | [-2.6, +4.4] | 9 / 7 |
| ruling_seen | +8.3 | [-2.1, +18.8] | 6 / 2 |
| statute | +0.0 | [+0.0, +0.0] | 0 / 0 |
| noanswer | +0.0 | [+0.0, +0.0] | 0 / 0 |
| calc | +0.0 | [+0.0, +0.0] | 0 / 0 |

H2 and H3 are the confirmatory tests (unseen-ruling row). All other rows are exploratory.
