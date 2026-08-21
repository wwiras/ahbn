#!/usr/bin/env python3
"""Validate and aggregate the final Stage 4 Exp08 rerun for E5."""
import argparse, hashlib, math
from datetime import datetime
from pathlib import Path
import pandas as pd
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
FINAL=(ROOT/'outputs/csv/exp08_results_20260821_164541.csv').resolve()
METRICS=['delivery_ratio','propagation_delay','duplicates','total_forwards']
NAMES={'gossip':'Gossip','cluster':'Structured','dcsoc':'DC-SoC','ahbn':'AHBN'}
SEEDS=set(range(42,62)); OVERLOADS={1.0,1.5,2.0,3.0}
def req(x,m):
    if not x: raise ValueError(m)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=FINAL)
    ap.add_argument('--timestamp',default=datetime.now().strftime('%Y%m%d_%H%M%S')); a=ap.parse_args()
    source=a.input.resolve(); req(source==FINAL,f'wrong input; required {FINAL}, got {source}')
    req(source.is_file(),f'missing input: {source}'); before=digest(source); df=pd.read_csv(source)
    print('E5 Exp08 final aggregation'); print(f'Input: {source}'); print(f'Input SHA-256: {before}')
    req(len(df)==320,f'expected 320 rows, found {len(df)}'); req(set(df.strategy)==set(NAMES),'comparator mismatch')
    req(set(df.seed)==SEEDS,'seeds are not exactly 42..61')
    df.ch_overload_factor=pd.to_numeric(df.ch_overload_factor,errors='coerce')
    req(df.ch_overload_factor.notna().all() and set(df.ch_overload_factor)==OVERLOADS,'overload mismatch/malformed')
    keys=['strategy','ch_overload_factor','seed']; req(not df.duplicated(keys).any(),'duplicate run identities')
    expected={(s,o,z) for s in NAMES for o in OVERLOADS for z in SEEDS}
    req(set(map(tuple,df[keys].itertuples(index=False,name=None)))==expected,'incomplete/extra run grid')
    for m in METRICS:
        req(m in df,f'missing {m}'); df[m]=pd.to_numeric(df[m],errors='coerce')
        req(df[m].notna().all() and df[m].map(math.isfinite).all(),f'invalid {m}')
    counts=df.groupby(['strategy','ch_overload_factor']).size(); req(len(counts)==16 and (counts==20).all(),'bad cells')
    rows=[]
    for (s,o),g in df.groupby(['strategy','ch_overload_factor'],sort=True):
        r={'comparator':NAMES[s],'strategy':s,'overload_factor':o,'n':len(g),'df':len(g)-1}
        for m in METRICS:
            mean=g[m].mean(); sd=g[m].std(ddof=1); se=sd/math.sqrt(len(g)); ci=t.ppf(.975,len(g)-1)*se
            r.update({f'{m}_mean':mean,f'{m}_sd':sd,f'{m}_se':se,f'{m}_ci95':ci,
                      f'{m}_ci95_low':mean-ci,f'{m}_ci95_high':mean+ci})
        rows.append(r)
    out=pd.DataFrame(rows); req(len(out)==16 and (out.n==20).all(),'aggregate validation failed')
    dest=ROOT/'outputs/csv'/f'exp08_final_summary_{a.timestamp}.csv'; out.to_csv(dest,index=False)
    req(before==digest(source),'raw input changed during aggregation')
    print('Raw rows: 320\nComparators: 4\nOverload factors: 4\nConditions: 16\nRuns per condition: 20')
    print('Seeds: 42..61 per condition\nDuplicate identities: 0\nInvalid required metrics: 0')
    print('95% CI: two-sided Student t; df=19'); print(f'Saved: {dest}'); print('E5 RESULT: PASS')
if __name__=='__main__': main()
