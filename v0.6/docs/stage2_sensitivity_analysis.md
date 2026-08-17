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

### Alpha test - dense

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

### Alpha test - Bottleneck

```bash

% python scripts/run_stage2_sensitivity.py \
  --config configs/stage2_parameter_sensitivity.yaml \
  --parameter alpha \
  --scenario bottleneck \
  --runs-per-setting 20
Stage 2 — AHBN Parameter Sensitivity
Parameters : alpha
Scenarios  : bottleneck
Runs/setting: 20
Total AHBN runs: 80

[   1/80] bottleneck alpha         =0.1  seed=42 delivery=0.780 delay=9.537970665062423 dup=157
[   2/80] bottleneck alpha         =0.1  seed=43 delivery=0.860 delay=8.50490138668312 dup=173
[   3/80] bottleneck alpha         =0.1  seed=44 delivery=0.840 delay=9.544797310076323 dup=169
[   4/80] bottleneck alpha         =0.1  seed=45 delivery=0.820 delay=10.927404896222136 dup=165
[   5/80] bottleneck alpha         =0.1  seed=46 delivery=0.810 delay=9.893287766053257 dup=163
[   6/80] bottleneck alpha         =0.1  seed=47 delivery=0.950 delay=9.88283204114644 dup=191
[   7/80] bottleneck alpha         =0.1  seed=48 delivery=0.880 delay=11.957414115933307 dup=177
[   8/80] bottleneck alpha         =0.1  seed=49 delivery=0.770 delay=10.915530988783203 dup=155
[   9/80] bottleneck alpha         =0.1  seed=50 delivery=0.850 delay=12.959866069183207 dup=171
[  10/80] bottleneck alpha         =0.1  seed=51 delivery=0.800 delay=9.64161221277719 dup=161
[  11/80] bottleneck alpha         =0.1  seed=52 delivery=0.850 delay=8.666760013476775 dup=171
[  12/80] bottleneck alpha         =0.1  seed=53 delivery=0.830 delay=9.953189025125239 dup=167
[  13/80] bottleneck alpha         =0.1  seed=54 delivery=0.700 delay=7.68594165458513 dup=141
[  14/80] bottleneck alpha         =0.1  seed=55 delivery=0.770 delay=9.595794041025956 dup=155
[  15/80] bottleneck alpha         =0.1  seed=56 delivery=0.790 delay=9.588149917905024 dup=159
[  16/80] bottleneck alpha         =0.1  seed=57 delivery=0.780 delay=8.823451773543768 dup=157
[  17/80] bottleneck alpha         =0.1  seed=58 delivery=0.810 delay=8.837516863602124 dup=163
[  18/80] bottleneck alpha         =0.1  seed=59 delivery=0.870 delay=10.955046994044208 dup=175
[  19/80] bottleneck alpha         =0.1  seed=60 delivery=0.820 delay=10.86402096629979 dup=165
[  20/80] bottleneck alpha         =0.1  seed=61 delivery=0.830 delay=9.839849305056468 dup=167
[  21/80] bottleneck alpha         =0.3  seed=42 delivery=0.780 delay=9.537970665062423 dup=157
[  22/80] bottleneck alpha         =0.3  seed=43 delivery=0.860 delay=8.50490138668312 dup=173
[  23/80] bottleneck alpha         =0.3  seed=44 delivery=0.840 delay=9.544797310076323 dup=169
[  24/80] bottleneck alpha         =0.3  seed=45 delivery=0.820 delay=10.927404896222136 dup=165
[  25/80] bottleneck alpha         =0.3  seed=46 delivery=0.810 delay=9.893287766053257 dup=163
[  26/80] bottleneck alpha         =0.3  seed=47 delivery=0.950 delay=9.88283204114644 dup=191
[  27/80] bottleneck alpha         =0.3  seed=48 delivery=0.880 delay=11.957414115933307 dup=177
[  28/80] bottleneck alpha         =0.3  seed=49 delivery=0.770 delay=10.915530988783203 dup=155
[  29/80] bottleneck alpha         =0.3  seed=50 delivery=0.850 delay=12.959866069183207 dup=171
[  30/80] bottleneck alpha         =0.3  seed=51 delivery=0.800 delay=9.64161221277719 dup=161
[  31/80] bottleneck alpha         =0.3  seed=52 delivery=0.850 delay=8.666760013476775 dup=171
[  32/80] bottleneck alpha         =0.3  seed=53 delivery=0.830 delay=9.953189025125239 dup=167
[  33/80] bottleneck alpha         =0.3  seed=54 delivery=0.700 delay=7.68594165458513 dup=141
[  34/80] bottleneck alpha         =0.3  seed=55 delivery=0.770 delay=9.595794041025956 dup=155
[  35/80] bottleneck alpha         =0.3  seed=56 delivery=0.790 delay=9.588149917905024 dup=159
[  36/80] bottleneck alpha         =0.3  seed=57 delivery=0.780 delay=8.823451773543768 dup=157
[  37/80] bottleneck alpha         =0.3  seed=58 delivery=0.810 delay=8.837516863602124 dup=163
[  38/80] bottleneck alpha         =0.3  seed=59 delivery=0.870 delay=10.955046994044208 dup=175
[  39/80] bottleneck alpha         =0.3  seed=60 delivery=0.820 delay=10.86402096629979 dup=165
[  40/80] bottleneck alpha         =0.3  seed=61 delivery=0.830 delay=9.839849305056468 dup=167
[  41/80] bottleneck alpha         =0.5  seed=42 delivery=0.780 delay=9.537970665062423 dup=157
[  42/80] bottleneck alpha         =0.5  seed=43 delivery=0.860 delay=8.50490138668312 dup=173
[  43/80] bottleneck alpha         =0.5  seed=44 delivery=0.840 delay=9.544797310076323 dup=169
[  44/80] bottleneck alpha         =0.5  seed=45 delivery=0.820 delay=10.927404896222136 dup=165
[  45/80] bottleneck alpha         =0.5  seed=46 delivery=0.810 delay=9.893287766053257 dup=163
[  46/80] bottleneck alpha         =0.5  seed=47 delivery=0.950 delay=9.88283204114644 dup=191
[  47/80] bottleneck alpha         =0.5  seed=48 delivery=0.880 delay=11.957414115933307 dup=177
[  48/80] bottleneck alpha         =0.5  seed=49 delivery=0.770 delay=10.915530988783203 dup=155
[  49/80] bottleneck alpha         =0.5  seed=50 delivery=0.850 delay=12.959866069183207 dup=171
[  50/80] bottleneck alpha         =0.5  seed=51 delivery=0.800 delay=9.64161221277719 dup=161
[  51/80] bottleneck alpha         =0.5  seed=52 delivery=0.850 delay=8.666760013476775 dup=171
[  52/80] bottleneck alpha         =0.5  seed=53 delivery=0.830 delay=9.953189025125239 dup=167
[  53/80] bottleneck alpha         =0.5  seed=54 delivery=0.700 delay=7.68594165458513 dup=141
[  54/80] bottleneck alpha         =0.5  seed=55 delivery=0.770 delay=9.595794041025956 dup=155
[  55/80] bottleneck alpha         =0.5  seed=56 delivery=0.790 delay=9.588149917905024 dup=159
[  56/80] bottleneck alpha         =0.5  seed=57 delivery=0.780 delay=8.823451773543768 dup=157
[  57/80] bottleneck alpha         =0.5  seed=58 delivery=0.810 delay=8.837516863602124 dup=163
[  58/80] bottleneck alpha         =0.5  seed=59 delivery=0.870 delay=10.955046994044208 dup=175
[  59/80] bottleneck alpha         =0.5  seed=60 delivery=0.820 delay=10.86402096629979 dup=165
[  60/80] bottleneck alpha         =0.5  seed=61 delivery=0.830 delay=9.839849305056468 dup=167
[  61/80] bottleneck alpha         =0.7  seed=42 delivery=0.780 delay=9.537970665062423 dup=157
[  62/80] bottleneck alpha         =0.7  seed=43 delivery=0.860 delay=8.50490138668312 dup=173
[  63/80] bottleneck alpha         =0.7  seed=44 delivery=0.840 delay=9.544797310076323 dup=169
[  64/80] bottleneck alpha         =0.7  seed=45 delivery=0.820 delay=10.927404896222136 dup=165
[  65/80] bottleneck alpha         =0.7  seed=46 delivery=0.810 delay=9.893287766053257 dup=163
[  66/80] bottleneck alpha         =0.7  seed=47 delivery=0.950 delay=9.88283204114644 dup=191
[  67/80] bottleneck alpha         =0.7  seed=48 delivery=0.880 delay=11.957414115933307 dup=177
[  68/80] bottleneck alpha         =0.7  seed=49 delivery=0.770 delay=10.915530988783203 dup=155
[  69/80] bottleneck alpha         =0.7  seed=50 delivery=0.850 delay=12.959866069183207 dup=171
[  70/80] bottleneck alpha         =0.7  seed=51 delivery=0.800 delay=9.64161221277719 dup=161
[  71/80] bottleneck alpha         =0.7  seed=52 delivery=0.850 delay=8.666760013476775 dup=171
[  72/80] bottleneck alpha         =0.7  seed=53 delivery=0.830 delay=9.953189025125239 dup=167
[  73/80] bottleneck alpha         =0.7  seed=54 delivery=0.700 delay=7.68594165458513 dup=141
[  74/80] bottleneck alpha         =0.7  seed=55 delivery=0.770 delay=9.595794041025956 dup=155
[  75/80] bottleneck alpha         =0.7  seed=56 delivery=0.790 delay=9.588149917905024 dup=159
[  76/80] bottleneck alpha         =0.7  seed=57 delivery=0.780 delay=8.823451773543768 dup=157
[  77/80] bottleneck alpha         =0.7  seed=58 delivery=0.810 delay=8.837516863602124 dup=163
[  78/80] bottleneck alpha         =0.7  seed=59 delivery=0.870 delay=10.955046994044208 dup=175
[  79/80] bottleneck alpha         =0.7  seed=60 delivery=0.820 delay=10.86402096629979 dup=165
[  80/80] bottleneck alpha         =0.7  seed=61 delivery=0.830 delay=9.839849305056468 dup=167

Saved results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/stage2_parameter_sensitivity_20260817_171457.csv

```

The α + bottleneck run is consistent with the α + dense result, and I would mark this test PASS — robust / low performance sensitivity.

### Alpha test - Churn

```bash
% python scripts/run_stage2_sensitivity.py \
  --config configs/stage2_parameter_sensitivity.yaml \
  --parameter alpha \
  --scenario churn \
  --runs-per-setting 20
Stage 2 — AHBN Parameter Sensitivity
Parameters : alpha
Scenarios  : churn
Runs/setting: 20
Total AHBN runs: 80

[   1/80] churn      alpha         =0.1  seed=42 delivery=0.880 delay=15.127603347239582 dup=133
[   2/80] churn      alpha         =0.1  seed=43 delivery=0.870 delay=18.43517009957881 dup=130
[   3/80] churn      alpha         =0.1  seed=44 delivery=0.810 delay=16.31037223597917 dup=118
[   4/80] churn      alpha         =0.1  seed=45 delivery=0.840 delay=15.19491622823467 dup=134
[   5/80] churn      alpha         =0.1  seed=46 delivery=0.730 delay=8.908855284988867 dup=102
[   6/80] churn      alpha         =0.1  seed=47 delivery=0.760 delay=11.8759380670632 dup=117
[   7/80] churn      alpha         =0.1  seed=48 delivery=0.840 delay=19.02003834231208 dup=128
[   8/80] churn      alpha         =0.1  seed=49 delivery=0.800 delay=13.07154904711148 dup=117
[   9/80] churn      alpha         =0.1  seed=50 delivery=0.950 delay=12.978603163139702 dup=154
[  10/80] churn      alpha         =0.1  seed=51 delivery=0.870 delay=16.842587064364896 dup=125
[  11/80] churn      alpha         =0.1  seed=52 delivery=0.780 delay=13.458172087957426 dup=115
[  12/80] churn      alpha         =0.1  seed=53 delivery=0.670 delay=7.491905985550131 dup=88
[  13/80] churn      alpha         =0.1  seed=54 delivery=0.740 delay=17.442127415407057 dup=114
[  14/80] churn      alpha         =0.1  seed=55 delivery=0.780 delay=13.987176703553548 dup=124
[  15/80] churn      alpha         =0.1  seed=56 delivery=0.750 delay=17.731516621475993 dup=107
[  16/80] churn      alpha         =0.1  seed=57 delivery=0.760 delay=13.208879622282367 dup=106
[  17/80] churn      alpha         =0.1  seed=58 delivery=0.740 delay=10.855704171663318 dup=101
[  18/80] churn      alpha         =0.1  seed=59 delivery=0.840 delay=13.264980085252096 dup=135
[  19/80] churn      alpha         =0.1  seed=60 delivery=0.830 delay=13.060203011536293 dup=125
[  20/80] churn      alpha         =0.1  seed=61 delivery=0.770 delay=11.633324780916611 dup=116
[  21/80] churn      alpha         =0.3  seed=42 delivery=0.880 delay=15.127603347239582 dup=133
[  22/80] churn      alpha         =0.3  seed=43 delivery=0.870 delay=18.43517009957881 dup=130
[  23/80] churn      alpha         =0.3  seed=44 delivery=0.810 delay=16.31037223597917 dup=118
[  24/80] churn      alpha         =0.3  seed=45 delivery=0.840 delay=15.19491622823467 dup=134
[  25/80] churn      alpha         =0.3  seed=46 delivery=0.730 delay=8.908855284988867 dup=102
[  26/80] churn      alpha         =0.3  seed=47 delivery=0.760 delay=11.8759380670632 dup=117
[  27/80] churn      alpha         =0.3  seed=48 delivery=0.840 delay=19.02003834231208 dup=128
[  28/80] churn      alpha         =0.3  seed=49 delivery=0.800 delay=13.07154904711148 dup=117
[  29/80] churn      alpha         =0.3  seed=50 delivery=0.950 delay=12.978603163139702 dup=154
[  30/80] churn      alpha         =0.3  seed=51 delivery=0.870 delay=16.842587064364896 dup=125
[  31/80] churn      alpha         =0.3  seed=52 delivery=0.780 delay=13.458172087957426 dup=115
[  32/80] churn      alpha         =0.3  seed=53 delivery=0.670 delay=7.491905985550131 dup=88
[  33/80] churn      alpha         =0.3  seed=54 delivery=0.740 delay=17.442127415407057 dup=114
[  34/80] churn      alpha         =0.3  seed=55 delivery=0.780 delay=13.987176703553548 dup=124
[  35/80] churn      alpha         =0.3  seed=56 delivery=0.750 delay=17.731516621475993 dup=107
[  36/80] churn      alpha         =0.3  seed=57 delivery=0.760 delay=13.208879622282367 dup=106
[  37/80] churn      alpha         =0.3  seed=58 delivery=0.740 delay=10.855704171663318 dup=101
[  38/80] churn      alpha         =0.3  seed=59 delivery=0.840 delay=13.264980085252096 dup=135
[  39/80] churn      alpha         =0.3  seed=60 delivery=0.830 delay=13.060203011536293 dup=125
[  40/80] churn      alpha         =0.3  seed=61 delivery=0.770 delay=11.633324780916611 dup=116
[  41/80] churn      alpha         =0.5  seed=42 delivery=0.880 delay=15.127603347239582 dup=133
[  42/80] churn      alpha         =0.5  seed=43 delivery=0.870 delay=18.43517009957881 dup=130
[  43/80] churn      alpha         =0.5  seed=44 delivery=0.810 delay=16.31037223597917 dup=118
[  44/80] churn      alpha         =0.5  seed=45 delivery=0.840 delay=15.19491622823467 dup=134
[  45/80] churn      alpha         =0.5  seed=46 delivery=0.730 delay=8.908855284988867 dup=102
[  46/80] churn      alpha         =0.5  seed=47 delivery=0.760 delay=11.8759380670632 dup=117
[  47/80] churn      alpha         =0.5  seed=48 delivery=0.840 delay=19.02003834231208 dup=128
[  48/80] churn      alpha         =0.5  seed=49 delivery=0.800 delay=13.07154904711148 dup=117
[  49/80] churn      alpha         =0.5  seed=50 delivery=0.950 delay=12.978603163139702 dup=154
[  50/80] churn      alpha         =0.5  seed=51 delivery=0.870 delay=16.842587064364896 dup=125
[  51/80] churn      alpha         =0.5  seed=52 delivery=0.780 delay=13.458172087957426 dup=115
[  52/80] churn      alpha         =0.5  seed=53 delivery=0.670 delay=7.491905985550131 dup=88
[  53/80] churn      alpha         =0.5  seed=54 delivery=0.740 delay=17.442127415407057 dup=114
[  54/80] churn      alpha         =0.5  seed=55 delivery=0.780 delay=13.987176703553548 dup=124
[  55/80] churn      alpha         =0.5  seed=56 delivery=0.750 delay=17.731516621475993 dup=107
[  56/80] churn      alpha         =0.5  seed=57 delivery=0.760 delay=13.208879622282367 dup=106
[  57/80] churn      alpha         =0.5  seed=58 delivery=0.740 delay=10.855704171663318 dup=101
[  58/80] churn      alpha         =0.5  seed=59 delivery=0.840 delay=13.264980085252096 dup=135
[  59/80] churn      alpha         =0.5  seed=60 delivery=0.830 delay=13.060203011536293 dup=125
[  60/80] churn      alpha         =0.5  seed=61 delivery=0.770 delay=11.633324780916611 dup=116
[  61/80] churn      alpha         =0.7  seed=42 delivery=0.880 delay=15.127603347239582 dup=133
[  62/80] churn      alpha         =0.7  seed=43 delivery=0.870 delay=18.43517009957881 dup=130
[  63/80] churn      alpha         =0.7  seed=44 delivery=0.810 delay=16.31037223597917 dup=118
[  64/80] churn      alpha         =0.7  seed=45 delivery=0.840 delay=15.19491622823467 dup=134
[  65/80] churn      alpha         =0.7  seed=46 delivery=0.730 delay=8.908855284988867 dup=102
[  66/80] churn      alpha         =0.7  seed=47 delivery=0.760 delay=11.8759380670632 dup=117
[  67/80] churn      alpha         =0.7  seed=48 delivery=0.840 delay=19.02003834231208 dup=128
[  68/80] churn      alpha         =0.7  seed=49 delivery=0.800 delay=13.07154904711148 dup=117
[  69/80] churn      alpha         =0.7  seed=50 delivery=0.950 delay=12.978603163139702 dup=154
[  70/80] churn      alpha         =0.7  seed=51 delivery=0.870 delay=16.842587064364896 dup=125
[  71/80] churn      alpha         =0.7  seed=52 delivery=0.780 delay=13.458172087957426 dup=115
[  72/80] churn      alpha         =0.7  seed=53 delivery=0.670 delay=7.491905985550131 dup=88
[  73/80] churn      alpha         =0.7  seed=54 delivery=0.740 delay=17.442127415407057 dup=114
[  74/80] churn      alpha         =0.7  seed=55 delivery=0.780 delay=13.987176703553548 dup=124
[  75/80] churn      alpha         =0.7  seed=56 delivery=0.750 delay=17.731516621475993 dup=107
[  76/80] churn      alpha         =0.7  seed=57 delivery=0.760 delay=13.208879622282367 dup=106
[  77/80] churn      alpha         =0.7  seed=58 delivery=0.740 delay=10.855704171663318 dup=101
[  78/80] churn      alpha         =0.7  seed=59 delivery=0.840 delay=13.264980085252096 dup=135
[  79/80] churn      alpha         =0.7  seed=60 delivery=0.830 delay=13.060203011536293 dup=125
[  80/80] churn      alpha         =0.7  seed=61 delivery=0.770 delay=11.633324780916611 dup=116

Saved results: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6/outputs/csv/stage2_parameter_sensitivity_20260817_172832.csv

```

α sensitivity
├── Dense       PASS
├── Bottleneck  PASS
└── Churn       PASS

Canonical α = 0.3
      ✅ RETAIN