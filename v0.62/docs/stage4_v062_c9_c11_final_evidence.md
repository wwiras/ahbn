# Control Simulator v0.62 — C9/C10/C11 final evidence

Frozen on: `2026-08-26T00:34:39.404613+00:00`. Parent v0.61 is preserved as historical pre-canonical-correction evidence; v0.62 is authoritative post-canonical-freeze Control Simulator evidence. No simulations or inferential tests were run during C9-C11.

## C9 — quantitative evidence

S5 PASS: exactly 840 formal rows (Exp07 120, Exp08 320, Exp09 400), 840 unique run keys, seeds 42–61, no missing cells, duplicate runs, malformed metrics, or smoke contamination. Exp07 has five Gossip fanout cells and one canonical adaptive AHBN cell. S6 PASS: 42 cells, every n=20, sample SD and Student-t 95% CI (`df=19`). Independent direct-raw verification covered 19 required cells and 304 scalar values; maximum absolute discrepancy was `1.1368683772161603e-13`. S7 numerical verification checked 90/90 plotted records with zero mismatch.

S7 visual readability PASS. The three saved PNGs were opened and inspected: legend overlap NONE; axis/tick-label overlap NONE; clipped text/error bars NONE; confidence intervals and marker identities readable. Exp07 shows AHBN once, as a canonical adaptive marker, not a fanout sweep.

## C10 — Exp07 results and interpretation

| Algorithm | Condition | n | Delivery ratio | Propagation delay | Duplicates | Total forwards |
|---|---|---:|---:|---:|---:|---:|
| Gossip | `gossip_k=2` | 20 | 0.732 [0.694695, 0.769305] | 11.7258 [11.1601, 12.2915] | 74.2 [70.4695, 77.9305] | 146.4 [138.939, 153.861] |
| Gossip | `gossip_k=3` | 20 | 0.91 [0.894293, 0.925707] | 9.25238 [8.66477, 9.83999] | 153.15 [149.717, 156.583] | 243.15 [238.293, 248.007] |
| Gossip | `gossip_k=4` | 20 | 0.96 [0.950889, 0.969111] | 7.4629 [7.01109, 7.91471] | 201 [197.244, 204.756] | 296 [291.473, 300.527] |
| Gossip | `gossip_k=5` | 20 | 0.97 [0.960889, 0.979111] | 6.47846 [6.06064, 6.89629] | 233.25 [229.207, 237.293] | 329.25 [324.451, 334.049] |
| Gossip | `gossip_k=6` | 20 | 0.9855 [0.978629, 0.992371] | 5.58927 [5.36823, 5.81031] | 258.35 [254.553, 262.147] | 355.9 [351.63, 360.17] |
| AHBN | `ahbn_canonical_adaptive` | 20 | 0.91 [0.894293, 0.925707] | 9.25238 [8.66477, 9.83999] | 153.15 [149.717, 156.583] | 243.15 [238.293, 248.007] |

Increasing Gossip fanout moves along a clear observed tradeoff: delivery rises from 0.732 to 0.9855 and delay falls from 11.7258 to 5.58927, while duplicates rise from 74.2 to 258.35 and forwards from 146.4 to 355.9. AHBN's descriptive results coincide with Gossip k=3 in this experiment, but it is one adaptive condition, not a fixed-k sweep point. It therefore offers the k=3-level observed balance here: better delivery and delay than k=2, but lower delivery and higher delay than k=4–6; its duplicate/forward cost is higher than k=2 and lower than k=4–6.

| Condition | Rows | Gossip / Cluster | Fanout 2 / 3 / 4 | mean d_hat / l_hat / u_hat | z mean [q05, q95] | Mode / fanout transitions |
|---|---:|---:|---:|---:|---:|---:|
| `adaptive` | 4883 | 4062 / 821 | 241 / 4642 / 0 | 0.2554 / 0.2752 / 0.0803 | 0.1001 [-0.2459, 0.2203] | 180 / 54 |

The trace shows adaptation through both mode and fanout: 4,062 Gossip versus 821 Cluster decisions, fanout 2 on 241 rows and fanout 3 on 4,642 rows, with 180 recorded mode-switch flags and 54 fanout-change flags. Duplicate pressure contributes `-d` and can push z toward Cluster/lower fanout, while latency and utilization contribute `+l/+u`; the trace supports competing pressure, not a universal superiority claim.

## C10 — Exp08 results and interpretation

| Algorithm | Condition | n | Delivery ratio | Propagation delay | Duplicates | Total forwards |
|---|---|---:|---:|---:|---:|---:|
| Gossip | `ch_overload_factor=1.0` | 20 | 1 [1, 1] | 3.27922 [3.25153, 3.30691] | 384 [384, 384] | 483 [483, 483] |
| Structured | `ch_overload_factor=1.0` | 20 | 1 [1, 1] | 4.51979 [4.46775, 4.57183] | 0 [0, 0] | 99 [99, 99] |
| DC-SoC | `ch_overload_factor=1.0` | 20 | 1 [1, 1] | 1.19827 [1.1976, 1.19894] | 0 [0, 0] | 99 [99, 99] |
| AHBN | `ch_overload_factor=1.0` | 20 | 0.91 [0.894293, 0.925707] | 9.25238 [8.66477, 9.83999] | 153.15 [149.717, 156.583] | 243.15 [238.293, 248.007] |
| Gossip | `ch_overload_factor=1.5` | 20 | 1 [1, 1] | 3.27922 [3.25153, 3.30691] | 384 [384, 384] | 483 [483, 483] |
| Structured | `ch_overload_factor=1.5` | 20 | 1 [1, 1] | 6.01979 [5.96775, 6.07183] | 0 [0, 0] | 99 [99, 99] |
| DC-SoC | `ch_overload_factor=1.5` | 20 | 1 [1, 1] | 1.69827 [1.6976, 1.69894] | 0 [0, 0] | 99 [99, 99] |
| AHBN | `ch_overload_factor=1.5` | 20 | 0.899 [0.87998, 0.91802] | 9.67954 [8.95788, 10.4012] | 151.85 [147.733, 155.967] | 240.75 [234.805, 246.695] |
| Gossip | `ch_overload_factor=2.0` | 20 | 1 [1, 1] | 3.27922 [3.25153, 3.30691] | 384 [384, 384] | 483 [483, 483] |
| Structured | `ch_overload_factor=2.0` | 20 | 1 [1, 1] | 7.51979 [7.46775, 7.57183] | 0 [0, 0] | 99 [99, 99] |
| DC-SoC | `ch_overload_factor=2.0` | 20 | 1 [1, 1] | 2.19827 [2.1976, 2.19894] | 0 [0, 0] | 99 [99, 99] |
| AHBN | `ch_overload_factor=2.0` | 20 | 0.9035 [0.888679, 0.918321] | 9.00279 [8.48213, 9.52344] | 151.85 [148.218, 155.482] | 241.2 [236.151, 246.249] |
| Gossip | `ch_overload_factor=3.0` | 20 | 1 [1, 1] | 3.27922 [3.25153, 3.30691] | 384 [384, 384] | 483 [483, 483] |
| Structured | `ch_overload_factor=3.0` | 20 | 1 [1, 1] | 10.5198 [10.4678, 10.5718] | 0 [0, 0] | 99 [99, 99] |
| DC-SoC | `ch_overload_factor=3.0` | 20 | 1 [1, 1] | 3.19827 [3.1976, 3.19894] | 0 [0, 0] | 99 [99, 99] |
| AHBN | `ch_overload_factor=3.0` | 20 | 0.8935 [0.871221, 0.915779] | 10.0333 [9.31789, 10.7488] | 150.7 [146.142, 155.258] | 239.05 [232.325, 245.775] |

Endpoint changes (factor 1.0 → 3.0):

| Algorithm | Delivery | Delay | Duplicates | Total forwards |
|---|---:|---:|---:|---:|
| Gossip | +0 (+0.00%) | +0 (+0.00%) | +0 (+0.00%) | +0 (+0.00%) |
| Structured | +0 (+0.00%) | +6 (+132.75%) | +0 (percentage undefined: zero baseline) | +0 (+0.00%) |
| DC-SoC | +0 (+0.00%) | +2 (+166.91%) | +0 (percentage undefined: zero baseline) | +0 (+0.00%) |
| AHBN | -0.0165 (-1.81%) | +0.780963 (+8.44%) | -2.45 (-1.60%) | -4.1 (-1.69%) |

Gossip is unchanged in all four endpoint metrics. Structured and DC-SoC retain delivery 1.0 and zero duplicates, while delay rises 132.75% and 166.91%, respectively. AHBN delay rises 8.44%, delivery falls 1.81%, duplicates fall 1.60%, and forwards fall 1.69%; intermediate AHBN means are non-monotonic. AHBN is neither the lowest-delay nor highest-delivery method here.

| Condition | Rows | Gossip / Cluster | Fanout 2 / 3 / 4 | mean d_hat / l_hat / u_hat | z mean [q05, q95] | Mode / fanout transitions |
|---|---:|---:|---:|---:|---:|---:|
| `ch_overload_factor=1.0` | 4883 | 4062 / 821 | 241 / 4642 / 0 | 0.2554 / 0.2752 / 0.0803 | 0.1001 [-0.2459, 0.2203] | 180 / 54 |
| `ch_overload_factor=1.5` | 4835 | 4102 / 733 | 147 / 4636 / 52 | 0.2536 / 0.2833 / 0.0807 | 0.1103 [-0.2047, 0.2225] | 167 / 138 |
| `ch_overload_factor=2.0` | 4844 | 4170 / 674 | 120 / 4626 / 98 | 0.2507 / 0.2872 / 0.0807 | 0.1171 [-0.1684, 0.2228] | 161 / 123 |
| `ch_overload_factor=3.0` | 4801 | 4158 / 643 | 119 / 4542 / 140 | 0.2567 / 0.2987 / 0.0805 | 0.1225 [-0.1586, 0.2246] | 161 / 121 |

From factor 1.0 to 3.0, mean `l_hat` rises 0.2752→0.2987, mean `u_hat` stays near 0.080, and mean `d_hat` is similar overall (0.2554→0.2567). Mean z rises 0.1001→0.1225, but competing terms make monotonic z unnecessary. Fanout 4 grows from 0 to 140 decisions and Cluster decisions decline from 821 to 643. Exp08 therefore activates both mode and fanout, predominantly fanout 3 with a growing high-score fanout-4 tail.

## C10 — Exp09 results and interpretation

| Algorithm | Condition | n | Delivery ratio | Propagation delay | Duplicates | Total forwards |
|---|---|---:|---:|---:|---:|---:|
| Gossip | `edge_prob=0.04` | 20 | 1 [1, 1] | 6.02986 [5.5476, 6.51212] | 209 [199.055, 218.945] | 306.6 [296.511, 316.689] |
| Structured | `edge_prob=0.04` | 20 | 1 [1, 1] | 4.52088 [4.47144, 4.57032] | 0 [0, 0] | 97.6 [97.1599, 98.0401] |
| DC-SoC | `edge_prob=0.04` | 20 | 1 [1, 1] | 1.19789 [1.19708, 1.1987] | 0 [0, 0] | 97.6 [97.1599, 98.0401] |
| AHBN | `edge_prob=0.04` | 20 | 0.961943 [0.952031, 0.971856] | 8.07568 [7.73807, 8.41329] | 131.9 [128.73, 135.07] | 225.75 [222.54, 228.96] |
| Gossip | `edge_prob=0.06` | 20 | 1 [1, 1] | 4.4425 [4.24607, 4.63893] | 408.5 [398.623, 418.377] | 507.35 [497.518, 517.182] |
| Structured | `edge_prob=0.06` | 20 | 1 [1, 1] | 4.51979 [4.46775, 4.57183] | 0 [0, 0] | 98.85 [98.6785, 99.0215] |
| DC-SoC | `edge_prob=0.06` | 20 | 1 [1, 1] | 1.19789 [1.19708, 1.1987] | 0 [0, 0] | 98.85 [98.6785, 99.0215] |
| AHBN | `edge_prob=0.06` | 20 | 0.966949 [0.958887, 0.975012] | 7.81458 [7.43547, 8.19369] | 179.2 [176.661, 181.739] | 274.75 [271.725, 277.775] |
| Gossip | `edge_prob=0.08` | 20 | 1 [1, 1] | 3.87189 [3.6707, 4.07308] | 601.9 [587.359, 616.441] | 700.9 [686.359, 715.441] |
| Structured | `edge_prob=0.08` | 20 | 1 [1, 1] | 4.51979 [4.46775, 4.57183] | 0 [0, 0] | 99 [99, 99] |
| DC-SoC | `edge_prob=0.08` | 20 | 1 [1, 1] | 1.19789 [1.19708, 1.1987] | 0 [0, 0] | 99 [99, 99] |
| AHBN | `edge_prob=0.08` | 20 | 0.9655 [0.957541, 0.973459] | 8.25732 [7.75056, 8.76409] | 190.7 [189.002, 192.398] | 286.25 [283.879, 288.621] |
| Gossip | `edge_prob=0.10` | 20 | 1 [1, 1] | 3.32364 [3.29309, 3.35419] | 792.8 [775.115, 810.485] | 891.8 [874.115, 909.485] |
| Structured | `edge_prob=0.10` | 20 | 1 [1, 1] | 4.51979 [4.46775, 4.57183] | 0 [0, 0] | 99 [99, 99] |
| DC-SoC | `edge_prob=0.10` | 20 | 1 [1, 1] | 1.19789 [1.19708, 1.1987] | 0 [0, 0] | 99 [99, 99] |
| AHBN | `edge_prob=0.10` | 20 | 0.9695 [0.961988, 0.977012] | 7.72429 [7.31239, 8.13618] | 194.45 [192.811, 196.089] | 290.4 [288.026, 292.774] |
| Gossip | `edge_prob=0.12` | 20 | 1 [1, 1] | 3.24195 [3.21125, 3.27266] | 984.1 [965.314, 1002.89] | 1083.1 [1064.31, 1101.89] |
| Structured | `edge_prob=0.12` | 20 | 1 [1, 1] | 4.51979 [4.46775, 4.57183] | 0 [0, 0] | 99 [99, 99] |
| DC-SoC | `edge_prob=0.12` | 20 | 1 [1, 1] | 1.19789 [1.19708, 1.1987] | 0 [0, 0] | 99 [99, 99] |
| AHBN | `edge_prob=0.12` | 20 | 0.9675 [0.960398, 0.974602] | 7.71413 [7.3228, 8.10545] | 194.3 [192.924, 195.676] | 290.05 [287.971, 292.129] |

Endpoint changes (p=0.04 → 0.12):

| Algorithm | Delivery | Delay | Duplicates | Total forwards |
|---|---:|---:|---:|---:|
| Gossip | +0 (+0.00%) | -2.7879 (-46.24%) | +775.1 (+370.86%) | +776.5 (+253.26%) |
| Structured | +0 (+0.00%) | -0.0010876 (-0.02%) | +0 (percentage undefined: zero baseline) | +1.4 (+1.43%) |
| DC-SoC | +0 (+0.00%) | +0 (+0.00%) | +0 (percentage undefined: zero baseline) | +1.4 (+1.43%) |
| AHBN | +0.00555674 (+0.58%) | -0.361556 (-4.48%) | +62.4 (+47.31%) | +64.3 (+28.48%) |

Gossip keeps delivery 1.0 while delay falls 46.24%, duplicates rise 370.86%, and forwards rise 253.26%. Structured and DC-SoC keep delivery 1.0 and zero duplicates; their forwards rise 1.43% as reachable topology size changes slightly. AHBN delivery rises 0.58%, delay falls 4.48%, duplicates rise 47.31%, and forwards rise 28.48%. AHBN has fewer duplicates but higher delay than Gossip at every tested density; no universal ranking is implied.

| Condition | Rows | Gossip / Cluster | Fanout 2 / 3 / 4 | mean d_hat / l_hat / u_hat | z mean [q05, q95] | Mode / fanout transitions |
|---|---:|---:|---:|---:|---:|---:|
| `edge_prob=0.04` | 4535 | 4467 / 68 | 0 / 4535 / 0 | 0.1538 / 0.2389 / 0.0739 | 0.1590 [0.0951, 0.2209] | 55 / 0 |
| `edge_prob=0.06` | 5515 | 5365 / 150 | 0 / 5515 / 0 | 0.1899 / 0.2568 / 0.0885 | 0.1554 [0.0282, 0.2216] | 97 / 0 |
| `edge_prob=0.08` | 5745 | 5537 / 208 | 2 / 5743 / 0 | 0.2011 / 0.2618 / 0.0918 | 0.1524 [0.0241, 0.2211] | 132 / 1 |
| `edge_prob=0.1` | 5828 | 5605 / 223 | 0 / 5828 / 0 | 0.2027 / 0.2625 / 0.0923 | 0.1522 [0.0233, 0.2215] | 144 / 0 |
| `edge_prob=0.12` | 5821 | 5614 / 207 | 1 / 5820 / 0 | 0.2014 / 0.2619 / 0.0924 | 0.1529 [0.0235, 0.2217] | 136 / 1 |

Mean `d_hat` rises from 0.1538 to about 0.2014 as density increases, strengthening `-d`; mean `l_hat` and `u_hat` also rise and counteract it. Mean z consequently changes only modestly and non-monotonically. AHBN remains overwhelmingly at fanout 3: two fanout-2 rows at p=0.08 and one at p=0.12, no fanout 4. Density response is therefore mainly mode adaptation, with extremely rare lower-fanout activation.

## Cross-experiment interpretation

v0.62 shows environment-specific controller use. Exp07 uses both mode and fanout around a moderate-cost operating point. Exp08 uses both, including an increasing fanout-4 tail under greater latency pressure. Exp09 responds mainly through mode while fanout remains almost entirely 3. Across experiments AHBN trades redundancy, reach, and delay; it does not dominate all comparators or metrics.

## C11 — v0.61 reconciliation

| Evidence/claim | v0.61 status | v0.62 result | Action |
|---|---|---|---|
| Exp07 descriptive means/CIs | Same numerical summary | Same means/CIs; adaptive trace proves fanout 2 and 3 operation | KEEP numbers; REWRITE fixed-fanout interpretation |
| Exp07 AHBN runtime fanout | Claimed fanout 3/no movement | 241 fanout-2 rows, 4,642 fanout-3 rows, 54 change flags | REWRITE |
| Exp08 endpoint percentages | 8.44% AHBN delay; 132.75% Structured; 166.91% DC-SoC | Confirmed; add all four metrics | KEEP and EXTEND |
| Exp08 AHBN fanout | Historical interpretation said fixed fanout 3 | fanout 4 occurs 52/98/140 times at factors 1.5/2.0/3.0 | REWRITE |
| Exp09 endpoints | +47.31% AHBN duplicates; -4.48% delay | Confirmed; delivery +0.58%, forwards +28.48% added | KEEP and EXTEND |
| Exp09 AHBN fanout | Historical interpretation said no reduction/fixed 3 | three formal fanout-2 rows and two change flags | REWRITE |
| Five-point AHBN Exp07 sweep | Unsupported | One canonical adaptive condition | REMOVE |
| Universal/significance language | Unsupported | Descriptive mean ± t CI only | REMOVE |

## Authoritative evidence and lineage

| Evidence | Path | SHA-256 |
|---|---|---|
| Exp07 results | `outputs/csv/exp07_results_20260826_081046.csv` | `766d4947839e27e11bcfc9989fa66ada74f5dba154840623507e48082b1f6307` |
| Exp08 results | `outputs/csv/exp08_results_20260826_081147.csv` | `ab602a0e1d6468ca0577df0899c5eaea7cf9a99d2e5b5e64425f4976a595682b` |
| Exp09 results | `outputs/csv/exp09_results_20260826_081323.csv` | `a5363eb521967355642ded4adedde63a46a1d1233c9780c786d34090838a44e6` |
| Exp07 trace | `outputs/csv/exp07_adaptive_trace_20260826_081047.csv` | `792f043d7b1f4668b745742b8d868d961dd7b356704fd6056fb3768f8ef1ff12` |
| Exp08 trace | `outputs/csv/exp08_ahbn_adaptive_trace_20260826_081147.csv` | `0c2e04d20a39c4fcb9626920a78ec2d1c41d94e255cc33aed7777ade5310849c` |
| Exp09 trace | `outputs/csv/exp09_adaptive_trace_20260826_081323.csv` | `12a5f86ed78d7e1306ea4cd3b5a6d94ece3f52fba928f92093dcbb0ab0910964` |
| Exp08 execution evidence | `outputs/csv/exp08_execution_evidence_20260826_081147.csv` | `974b00fa43b0a6d67242ef04f33afc35cb6bbcc82ae6a6e5548852ec77a6b7df` |
| Exp08 execution manifest | `outputs/csv/exp08_s3_manifest.json` | `d49d1bcfa96c8bd6cdff4881fdd2c4aa27899d44bec6190910c0d744f17a3d8b` |
| S5 aggregate | `outputs/csv/final_control_v062_s5_raw.csv` | `742819f6903c229dbc346b951caaa57daf87e059ec6710d7ab43950e2e1935a5` |
| S5 summary | `outputs/csv/final_control_v062_s5_summary.csv` | `0271f723d471ab9ed8ec66fd8c224a6331074e6db4a3ababb9a5191c19b0103b` |
| S6 statistics | `outputs/csv/final_control_v062_s6_statistics.csv` | `a1f259b2a8727548639aba28f6bcfebf08ebf6d0718dc18ac10c0c79d3e63ca4` |
| S6 seed robustness | `outputs/csv/final_control_v062_s6_seed_robustness.csv` | `fba8fe55f74252a0d58a0f48b554d085adc41440c6f0ee57f4cdcf6f79c4ffda` |
| S7 plot data | `outputs/csv/final_control_v062_s7_plotdata.csv` | `f59e1103eda90044e3424e83f30955876d070b00cd691981f48743c7fa82cf73` |
| exp07_v062_final_fanout.pdf | `outputs/figures/v062_s7/exp07_v062_final_fanout.pdf` | `3d9df53b433c4623bd4e3ffb891ec859a1a85ce5e5f66b66d330b63a78f3f5a2` |
| exp07_v062_final_fanout.png | `outputs/figures/v062_s7/exp07_v062_final_fanout.png` | `155121baa0d79ff99b9fda2a53010d6c42db03b0b0fc0e7e736d171bf4b3ca65` |
| exp08_v062_final_overload.pdf | `outputs/figures/v062_s7/exp08_v062_final_overload.pdf` | `a4372b7e3c14a6869a04b6288ff42978e347934a54be04d5ccb0d3d1fb400d0c` |
| exp08_v062_final_overload.png | `outputs/figures/v062_s7/exp08_v062_final_overload.png` | `90f1720fb08c4c1bbf05cdefc5b676c0680ebefac79cbc89b2edf4c88cd2edb7` |
| exp09_v062_final_density.pdf | `outputs/figures/v062_s7/exp09_v062_final_density.pdf` | `c49bd4d53ef31fb510b3d66732cd5f5868cbda30deb76630873cdf590f351e9f` |
| exp09_v062_final_density.png | `outputs/figures/v062_s7/exp09_v062_final_density.png` | `dc4f6668c735a05155813c97741ff9dcd3f0b97bcf34d9fbf3692e215eba5ba7` |

Lineage is exact formal result CSVs → versioned S5 aggregate → versioned S6 statistics → versioned S7 plot data/figures → this C10 interpretation and the C11 manifest. Trace files support only controller-behaviour interpretation. Smoke files are excluded.

## Freeze gate

Canonical validator PASS; formal Exp07/08/09 PASS; 840/840 runs; AHBN invariant failures 0; S5/S6/S7 PASS; C10 complete; hashes recorded; smoke contamination NONE; post-formal algorithm tuning NONE; controller/comparator/config/raw modifications NONE. v0.61 remains historical and unmodified.
