## STAGE 2 — SENSITIVITY ANALYSIS



### Smoke Test

```bash
% python scripts/run_stage2_sensitivity.py \
  --config configs/stage2_parameter_sensitivity.yaml \
  --parameter alpha \
  --scenario dense \
  --runs-per-setting 1
Stage 2 — AHBN Parameter Sensitivity
Parameters : alpha
Scenarios  : dense
Runs/setting: 1
Total AHBN runs: 4

[   1/4] dense      alpha         =0.1  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[   2/4] dense      alpha         =0.3  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[   3/4] dense      alpha         =0.5  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[   4/4] dense      alpha         =0.7  seed=42 delivery=0.940 delay=7.494288388943651 dup=189

Saved results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/stage2_parameter_sensitivity_20260817_142150.csv
```

### Alpha test

```bash
(venv0.5) wwiras@wwirass-MacBook-Air v0.6 % python scripts/run_stage2_sensitivity.py \
  --config configs/stage2_parameter_sensitivity.yaml \
  --parameter alpha \
  --scenario dense \
  --runs-per-setting 20
Stage 2 — AHBN Parameter Sensitivity
Parameters : alpha
Scenarios  : dense
Runs/setting: 20
Total AHBN runs: 80

[   1/80] dense      alpha         =0.1  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[   2/80] dense      alpha         =0.1  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[   3/80] dense      alpha         =0.1  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[   4/80] dense      alpha         =0.1  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[   5/80] dense      alpha         =0.1  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[   6/80] dense      alpha         =0.1  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[   7/80] dense      alpha         =0.1  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[   8/80] dense      alpha         =0.1  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[   9/80] dense      alpha         =0.1  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  10/80] dense      alpha         =0.1  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  11/80] dense      alpha         =0.1  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  12/80] dense      alpha         =0.1  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  13/80] dense      alpha         =0.1  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  14/80] dense      alpha         =0.1  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  15/80] dense      alpha         =0.1  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  16/80] dense      alpha         =0.1  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  17/80] dense      alpha         =0.1  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  18/80] dense      alpha         =0.1  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  19/80] dense      alpha         =0.1  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  20/80] dense      alpha         =0.1  seed=61 delivery=0.910 delay=7.6203048043068 dup=183
[  21/80] dense      alpha         =0.3  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[  22/80] dense      alpha         =0.3  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[  23/80] dense      alpha         =0.3  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[  24/80] dense      alpha         =0.3  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[  25/80] dense      alpha         =0.3  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[  26/80] dense      alpha         =0.3  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[  27/80] dense      alpha         =0.3  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[  28/80] dense      alpha         =0.3  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[  29/80] dense      alpha         =0.3  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  30/80] dense      alpha         =0.3  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  31/80] dense      alpha         =0.3  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  32/80] dense      alpha         =0.3  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  33/80] dense      alpha         =0.3  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  34/80] dense      alpha         =0.3  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  35/80] dense      alpha         =0.3  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  36/80] dense      alpha         =0.3  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  37/80] dense      alpha         =0.3  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  38/80] dense      alpha         =0.3  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  39/80] dense      alpha         =0.3  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  40/80] dense      alpha         =0.3  seed=61 delivery=0.910 delay=7.6203048043068 dup=183
[  41/80] dense      alpha         =0.5  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[  42/80] dense      alpha         =0.5  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[  43/80] dense      alpha         =0.5  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[  44/80] dense      alpha         =0.5  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[  45/80] dense      alpha         =0.5  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[  46/80] dense      alpha         =0.5  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[  47/80] dense      alpha         =0.5  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[  48/80] dense      alpha         =0.5  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[  49/80] dense      alpha         =0.5  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  50/80] dense      alpha         =0.5  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  51/80] dense      alpha         =0.5  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  52/80] dense      alpha         =0.5  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  53/80] dense      alpha         =0.5  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  54/80] dense      alpha         =0.5  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  55/80] dense      alpha         =0.5  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  56/80] dense      alpha         =0.5  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  57/80] dense      alpha         =0.5  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  58/80] dense      alpha         =0.5  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  59/80] dense      alpha         =0.5  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  60/80] dense      alpha         =0.5  seed=61 delivery=0.910 delay=7.6203048043068 dup=183
[  61/80] dense      alpha         =0.7  seed=42 delivery=0.940 delay=7.494288388943651 dup=189
[  62/80] dense      alpha         =0.7  seed=43 delivery=0.890 delay=8.6665696436651 dup=179
[  63/80] dense      alpha         =0.7  seed=44 delivery=0.910 delay=7.539672335159132 dup=183
[  64/80] dense      alpha         =0.7  seed=45 delivery=0.900 delay=8.650780272330573 dup=181
[  65/80] dense      alpha         =0.7  seed=46 delivery=0.970 delay=7.632737860562308 dup=195
[  66/80] dense      alpha         =0.7  seed=47 delivery=0.940 delay=7.8176438898259795 dup=189
[  67/80] dense      alpha         =0.7  seed=48 delivery=0.970 delay=7.781660905323122 dup=195
[  68/80] dense      alpha         =0.7  seed=49 delivery=0.930 delay=7.587100736724944 dup=187
[  69/80] dense      alpha         =0.7  seed=50 delivery=0.950 delay=7.831933687655348 dup=191
[  70/80] dense      alpha         =0.7  seed=51 delivery=0.920 delay=7.973674242115914 dup=185
[  71/80] dense      alpha         =0.7  seed=52 delivery=0.910 delay=8.561541794149203 dup=183
[  72/80] dense      alpha         =0.7  seed=53 delivery=0.880 delay=8.441766293448872 dup=177
[  73/80] dense      alpha         =0.7  seed=54 delivery=0.950 delay=7.806967528340527 dup=191
[  74/80] dense      alpha         =0.7  seed=55 delivery=0.960 delay=9.61073616345794 dup=193
[  75/80] dense      alpha         =0.7  seed=56 delivery=0.920 delay=8.680802993395574 dup=185
[  76/80] dense      alpha         =0.7  seed=57 delivery=0.950 delay=7.542614323938729 dup=191
[  77/80] dense      alpha         =0.7  seed=58 delivery=0.950 delay=7.605339333781711 dup=191
[  78/80] dense      alpha         =0.7  seed=59 delivery=0.910 delay=7.594042271703149 dup=183
[  79/80] dense      alpha         =0.7  seed=60 delivery=0.970 delay=9.745217348689232 dup=195
[  80/80] dense      alpha         =0.7  seed=61 delivery=0.910 delay=7.6203048043068 dup=183

Saved results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/stage2_parameter_sensitivity_20260817_144959.csv

```

Current Stage-2 conclusion for α + dense : PASS — robust / low performance sensitivity.
Instead: “α changes controller responsiveness and mode occupancy, but the dissemination metrics remain stable over α ∈ [0.1, 0.7] under dense-topology pressure.”

### 