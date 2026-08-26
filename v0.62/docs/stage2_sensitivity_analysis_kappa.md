## STAGE 2 — SENSITIVITY ANALYSIS - Kappa (Sigmoid steepness)

### Dense

```bash

% python scripts/run_stage2_sensitivity.py \
  --config configs/stage2_parameter_sensitivity.yaml \
  --parameter kappa \
  --scenario dense \
  --runs-per-setting 20
Stage 2 — AHBN Parameter Sensitivity
Parameters : kappa
Scenarios  : dense
Runs/setting: 20
Total AHBN runs: 80

[   1/80] dense      kappa         =0.5  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[   2/80] dense      kappa         =0.5  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[   3/80] dense      kappa         =0.5  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[   4/80] dense      kappa         =0.5  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[   5/80] dense      kappa         =0.5  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[   6/80] dense      kappa         =0.5  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[   7/80] dense      kappa         =0.5  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[   8/80] dense      kappa         =0.5  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[   9/80] dense      kappa         =0.5  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  10/80] dense      kappa         =0.5  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  11/80] dense      kappa         =0.5  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  12/80] dense      kappa         =0.5  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  13/80] dense      kappa         =0.5  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  14/80] dense      kappa         =0.5  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  15/80] dense      kappa         =0.5  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  16/80] dense      kappa         =0.5  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  17/80] dense      kappa         =0.5  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  18/80] dense      kappa         =0.5  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  19/80] dense      kappa         =0.5  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  20/80] dense      kappa         =0.5  seed=61 delivery=0.910 delay=7.6203048043068 dup=183
[  21/80] dense      kappa         =1.0  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[  22/80] dense      kappa         =1.0  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[  23/80] dense      kappa         =1.0  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[  24/80] dense      kappa         =1.0  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[  25/80] dense      kappa         =1.0  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[  26/80] dense      kappa         =1.0  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[  27/80] dense      kappa         =1.0  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[  28/80] dense      kappa         =1.0  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[  29/80] dense      kappa         =1.0  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  30/80] dense      kappa         =1.0  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  31/80] dense      kappa         =1.0  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  32/80] dense      kappa         =1.0  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  33/80] dense      kappa         =1.0  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  34/80] dense      kappa         =1.0  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  35/80] dense      kappa         =1.0  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  36/80] dense      kappa         =1.0  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  37/80] dense      kappa         =1.0  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  38/80] dense      kappa         =1.0  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  39/80] dense      kappa         =1.0  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  40/80] dense      kappa         =1.0  seed=61 delivery=0.910 delay=7.6203048043068 dup=183
[  41/80] dense      kappa         =2.0  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[  42/80] dense      kappa         =2.0  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[  43/80] dense      kappa         =2.0  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[  44/80] dense      kappa         =2.0  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[  45/80] dense      kappa         =2.0  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[  46/80] dense      kappa         =2.0  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[  47/80] dense      kappa         =2.0  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[  48/80] dense      kappa         =2.0  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[  49/80] dense      kappa         =2.0  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  50/80] dense      kappa         =2.0  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  51/80] dense      kappa         =2.0  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  52/80] dense      kappa         =2.0  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  53/80] dense      kappa         =2.0  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  54/80] dense      kappa         =2.0  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  55/80] dense      kappa         =2.0  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  56/80] dense      kappa         =2.0  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  57/80] dense      kappa         =2.0  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  58/80] dense      kappa         =2.0  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  59/80] dense      kappa         =2.0  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  60/80] dense      kappa         =2.0  seed=61 delivery=0.910 delay=7.6203048043068 dup=183
[  61/80] dense      kappa         =4.0  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[  62/80] dense      kappa         =4.0  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[  63/80] dense      kappa         =4.0  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[  64/80] dense      kappa         =4.0  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[  65/80] dense      kappa         =4.0  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[  66/80] dense      kappa         =4.0  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[  67/80] dense      kappa         =4.0  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[  68/80] dense      kappa         =4.0  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[  69/80] dense      kappa         =4.0  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  70/80] dense      kappa         =4.0  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  71/80] dense      kappa         =4.0  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  72/80] dense      kappa         =4.0  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  73/80] dense      kappa         =4.0  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  74/80] dense      kappa         =4.0  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  75/80] dense      kappa         =4.0  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  76/80] dense      kappa         =4.0  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  77/80] dense      kappa         =4.0  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  78/80] dense      kappa         =4.0  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  79/80] dense      kappa         =4.0  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  80/80] dense      kappa         =4.0  seed=61 delivery=0.910 delay=7.6203048043068 dup=183

Saved results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/stage2_parameter_sensitivity_20260817_181054.csv

```

PASS — canonical κ = 1.0 retained
The reasoning is stronger than simply saying "κ doesn't matter."
The correct interpretation is:
Across κ = 0.5–4.0, dense-scenario network outcomes remained stable. Increasing κ progressively amplified the controller weight response, as theoretically expected. Mode-selection proportions remained unchanged because the canonical mode threshold of 0.5 makes mode selection dependent on the sign of the controller score rather than sigmoid steepness. Only the aggressive κ = 4 setting produced appreciable fanout changes, reducing mean fanout from 3.0 to approximately 2.90, without improving delivery, latency, or duplication. Thus, κ = 1.0 provides a moderate response without unnecessary fanout sensitivity.

### Kappa test - Churn

```bash
% python scripts/run_stage2_sensitivity.py \
  --config configs/stage2_parameter_sensitivity.yaml \
  --parameter kappa \
  --scenario churn \
  --runs-per-setting 20
Stage 2 — AHBN Parameter Sensitivity
Parameters : kappa
Scenarios  : churn
Runs/setting: 20
Total AHBN runs: 80

[   1/80] churn      kappa         =0.5  seed=42 delivery=0.880 delay=15.127603347239582 dup=133
[   2/80] churn      kappa         =0.5  seed=43 delivery=0.870 delay=18.43517009957881 dup=130
[   3/80] churn      kappa         =0.5  seed=44 delivery=0.810 delay=16.31037223597917 dup=118
[   4/80] churn      kappa         =0.5  seed=45 delivery=0.840 delay=15.19491622823467 dup=134
[   5/80] churn      kappa         =0.5  seed=46 delivery=0.730 delay=8.908855284988867 dup=102
[   6/80] churn      kappa         =0.5  seed=47 delivery=0.760 delay=11.8759380670632 dup=117
[   7/80] churn      kappa         =0.5  seed=48 delivery=0.840 delay=19.02003834231208 dup=128
[   8/80] churn      kappa         =0.5  seed=49 delivery=0.800 delay=13.07154904711148 dup=117
[   9/80] churn      kappa         =0.5  seed=50 delivery=0.950 delay=12.978603163139702 dup=154
[  10/80] churn      kappa         =0.5  seed=51 delivery=0.870 delay=16.842587064364896 dup=125
[  11/80] churn      kappa         =0.5  seed=52 delivery=0.780 delay=13.458172087957426 dup=115
[  12/80] churn      kappa         =0.5  seed=53 delivery=0.670 delay=7.491905985550131 dup=88
[  13/80] churn      kappa         =0.5  seed=54 delivery=0.740 delay=17.442127415407057 dup=114
[  14/80] churn      kappa         =0.5  seed=55 delivery=0.780 delay=13.987176703553548 dup=124
[  15/80] churn      kappa         =0.5  seed=56 delivery=0.750 delay=17.731516621475993 dup=107
[  16/80] churn      kappa         =0.5  seed=57 delivery=0.760 delay=13.208879622282367 dup=106
[  17/80] churn      kappa         =0.5  seed=58 delivery=0.740 delay=10.855704171663318 dup=101
[  18/80] churn      kappa         =0.5  seed=59 delivery=0.840 delay=13.264980085252096 dup=135
[  19/80] churn      kappa         =0.5  seed=60 delivery=0.830 delay=13.060203011536293 dup=125
[  20/80] churn      kappa         =0.5  seed=61 delivery=0.770 delay=11.633324780916611 dup=116
[  21/80] churn      kappa         =1.0  seed=42 delivery=0.880 delay=15.127603347239582 dup=133
[  22/80] churn      kappa         =1.0  seed=43 delivery=0.870 delay=18.43517009957881 dup=130
[  23/80] churn      kappa         =1.0  seed=44 delivery=0.810 delay=16.31037223597917 dup=118
[  24/80] churn      kappa         =1.0  seed=45 delivery=0.840 delay=15.19491622823467 dup=134
[  25/80] churn      kappa         =1.0  seed=46 delivery=0.730 delay=8.908855284988867 dup=102
[  26/80] churn      kappa         =1.0  seed=47 delivery=0.760 delay=11.8759380670632 dup=117
[  27/80] churn      kappa         =1.0  seed=48 delivery=0.840 delay=19.02003834231208 dup=128
[  28/80] churn      kappa         =1.0  seed=49 delivery=0.800 delay=13.07154904711148 dup=117
[  29/80] churn      kappa         =1.0  seed=50 delivery=0.950 delay=12.978603163139702 dup=154
[  30/80] churn      kappa         =1.0  seed=51 delivery=0.870 delay=16.842587064364896 dup=125
[  31/80] churn      kappa         =1.0  seed=52 delivery=0.780 delay=13.458172087957426 dup=115
[  32/80] churn      kappa         =1.0  seed=53 delivery=0.670 delay=7.491905985550131 dup=88
[  33/80] churn      kappa         =1.0  seed=54 delivery=0.740 delay=17.442127415407057 dup=114
[  34/80] churn      kappa         =1.0  seed=55 delivery=0.780 delay=13.987176703553548 dup=124
[  35/80] churn      kappa         =1.0  seed=56 delivery=0.750 delay=17.731516621475993 dup=107
[  36/80] churn      kappa         =1.0  seed=57 delivery=0.760 delay=13.208879622282367 dup=106
[  37/80] churn      kappa         =1.0  seed=58 delivery=0.740 delay=10.855704171663318 dup=101
[  38/80] churn      kappa         =1.0  seed=59 delivery=0.840 delay=13.264980085252096 dup=135
[  39/80] churn      kappa         =1.0  seed=60 delivery=0.830 delay=13.060203011536293 dup=125
[  40/80] churn      kappa         =1.0  seed=61 delivery=0.770 delay=11.633324780916611 dup=116
[  41/80] churn      kappa         =2.0  seed=42 delivery=0.880 delay=15.127603347239582 dup=133
[  42/80] churn      kappa         =2.0  seed=43 delivery=0.870 delay=18.43517009957881 dup=130
[  43/80] churn      kappa         =2.0  seed=44 delivery=0.810 delay=16.31037223597917 dup=118
[  44/80] churn      kappa         =2.0  seed=45 delivery=0.840 delay=15.19491622823467 dup=134
[  45/80] churn      kappa         =2.0  seed=46 delivery=0.730 delay=8.908855284988867 dup=102
[  46/80] churn      kappa         =2.0  seed=47 delivery=0.760 delay=11.8759380670632 dup=117
[  47/80] churn      kappa         =2.0  seed=48 delivery=0.840 delay=19.02003834231208 dup=128
[  48/80] churn      kappa         =2.0  seed=49 delivery=0.800 delay=13.07154904711148 dup=117
[  49/80] churn      kappa         =2.0  seed=50 delivery=0.950 delay=12.978603163139702 dup=154
[  50/80] churn      kappa         =2.0  seed=51 delivery=0.870 delay=16.842587064364896 dup=125
[  51/80] churn      kappa         =2.0  seed=52 delivery=0.780 delay=13.458172087957426 dup=115
[  52/80] churn      kappa         =2.0  seed=53 delivery=0.670 delay=7.491905985550131 dup=88
[  53/80] churn      kappa         =2.0  seed=54 delivery=0.740 delay=17.442127415407057 dup=114
[  54/80] churn      kappa         =2.0  seed=55 delivery=0.780 delay=13.987176703553548 dup=124
[  55/80] churn      kappa         =2.0  seed=56 delivery=0.750 delay=17.731516621475993 dup=107
[  56/80] churn      kappa         =2.0  seed=57 delivery=0.760 delay=13.208879622282367 dup=106
[  57/80] churn      kappa         =2.0  seed=58 delivery=0.740 delay=10.855704171663318 dup=101
[  58/80] churn      kappa         =2.0  seed=59 delivery=0.840 delay=13.264980085252096 dup=135
[  59/80] churn      kappa         =2.0  seed=60 delivery=0.830 delay=13.060203011536293 dup=125
[  60/80] churn      kappa         =2.0  seed=61 delivery=0.770 delay=11.633324780916611 dup=116
[  61/80] churn      kappa         =4.0  seed=42 delivery=0.890 delay=12.084359494756901 dup=184
[  62/80] churn      kappa         =4.0  seed=43 delivery=0.940 delay=15.455341331055521 dup=203
[  63/80] churn      kappa         =4.0  seed=44 delivery=0.840 delay=10.936124082073082 dup=177
[  64/80] churn      kappa         =4.0  seed=45 delivery=0.820 delay=9.698633774244065 dup=188
[  65/80] churn      kappa         =4.0  seed=46 delivery=0.860 delay=12.871667786452512 dup=173
[  66/80] churn      kappa         =4.0  seed=47 delivery=0.890 delay=12.911390919669566 dup=200
[  67/80] churn      kappa         =4.0  seed=48 delivery=0.810 delay=9.880734843780024 dup=150
[  68/80] churn      kappa         =4.0  seed=49 delivery=0.840 delay=11.63339495310123 dup=169
[  69/80] churn      kappa         =4.0  seed=50 delivery=0.890 delay=12.324732095964553 dup=204
[  70/80] churn      kappa         =4.0  seed=51 delivery=0.860 delay=11.991108125084033 dup=171
[  71/80] churn      kappa         =4.0  seed=52 delivery=0.740 delay=11.185211984614046 dup=136
[  72/80] churn      kappa         =4.0  seed=53 delivery=0.850 delay=12.15882338930441 dup=189
[  73/80] churn      kappa         =4.0  seed=54 delivery=0.860 delay=11.014754456899185 dup=187
[  74/80] churn      kappa         =4.0  seed=55 delivery=0.930 delay=14.107534093357224 dup=213
[  75/80] churn      kappa         =4.0  seed=56 delivery=0.900 delay=11.99893224763503 dup=178
[  76/80] churn      kappa         =4.0  seed=57 delivery=0.900 delay=12.24424732351119 dup=190
[  77/80] churn      kappa         =4.0  seed=58 delivery=0.840 delay=12.978240937178875 dup=170
[  78/80] churn      kappa         =4.0  seed=59 delivery=0.900 delay=13.039221288883036 dup=187
[  79/80] churn      kappa         =4.0  seed=60 delivery=0.930 delay=13.042317181758202 dup=212
[  80/80] churn      kappa         =4.0  seed=61 delivery=0.770 delay=10.846414040520653 dup=154

Saved results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/stage2_parameter_sensitivity_20260817_203042.csv

```
Scientific justification

The combined dense and churn experiments demonstrate that κ behaves as intended as the sigmoid-response steepness parameter. Moderate values (0.5–2.0) alter controller confidence without materially changing forwarding decisions, whereas κ = 4 produces substantially more aggressive fanout adaptation. Under churn, this improves delivery and latency but increases duplicate transmissions by approximately 52%; under dense conditions, the stronger response provides no corresponding network-level benefit. Therefore, κ = 1.0 is retained as a moderate operating point that avoids excessive fanout sensitivity while preserving adaptive directionality.
One nuance: I would not claim κ=1 is statistically superior to κ=0.5 or 2—they produced identical network outcomes here. Our justification is instead that 1.0 is the natural moderate sigmoid scale, while the sensitivity experiment shows robustness around it and exposes the cost of aggressive κ=4.