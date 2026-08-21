#!/usr/bin/env python3
"""Generate four final Exp08 figures from the E5 summary only."""
import argparse, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
COMPS=['Gossip','Structured','DC-SoC','AHBN']; LEVELS=[1.0,1.5,2.0,3.0]
METRICS={'delivery_ratio':'Delivery ratio','propagation_delay':'Propagation delay (s)',
         'duplicates':'Duplicates','total_forwards':'Total forwards'}
STYLES=[('o','-'),('s','--'),('^','-.'),('D',':')]
def req(x,m):
    if not x: raise ValueError(m)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--summary',type=Path,required=True); ap.add_argument('--timestamp',required=True); a=ap.parse_args()
    source=a.summary.resolve(); req(source.parent==(ROOT/'outputs/csv').resolve() and source.name.startswith('exp08_final_summary_'),'not a final summary')
    df=pd.read_csv(source); required={'comparator','overload_factor','n'}
    for m in METRICS: required|={f'{m}_mean',f'{m}_ci95'}
    req(not(required-set(df)),f'missing columns {sorted(required-set(df))}')
    req(len(df)==16 and set(df.comparator)==set(COMPS) and set(df.overload_factor)==set(LEVELS),'invalid grid')
    req((df.n==20).all() and not df.duplicated(['comparator','overload_factor']).any(),'invalid cells')
    outputs=[]; colors=plt.rcParams['axes.prop_cycle'].by_key()['color'][:4]
    for metric,ylabel in METRICS.items():
        fig,ax=plt.subplots(figsize=(7.2,4.8),constrained_layout=True)
        for comp,color,(marker,line) in zip(COMPS,colors,STYLES):
            r=df[df.comparator==comp].sort_values('overload_factor'); req(len(r)==4,f'bad {comp} row count')
            mean=pd.to_numeric(r[f'{metric}_mean'],errors='coerce'); ci=pd.to_numeric(r[f'{metric}_ci95'],errors='coerce')
            req(mean.notna().all() and ci.notna().all() and mean.map(math.isfinite).all() and ci.map(math.isfinite).all() and (ci>=0).all(),f'invalid {metric}')
            ax.errorbar(r.overload_factor,mean,yerr=ci,label=comp,color=color,marker=marker,linestyle=line,linewidth=1.7,markersize=5.5,capsize=3,elinewidth=1.1)
        ax.set_xlabel('CH overload factor'); ax.set_ylabel(ylabel); ax.set_xticks(LEVELS); ax.grid(True,linestyle=':',linewidth=.7,alpha=.65); ax.legend(frameon=False,ncols=2)
        dest=ROOT/'outputs/figures'/f'exp08_final_{metric}_{a.timestamp}.png'; dest.parent.mkdir(parents=True,exist_ok=True); fig.savefig(dest,dpi=300,bbox_inches='tight'); plt.close(fig)
        req(dest.is_file() and dest.stat().st_size>0,f'missing {dest}'); outputs.append(dest)
    print('E6 Exp08 final plotting'); print(f'Summary input only: {source}'); print('Validation: 16 conditions; n=20; 4 comparators x 4 overloads'); print('Error bars: mean +/- Student-t 95% CI')
    for p in outputs: print(f'Saved: {p}')
    print('E6 RESULT: PASS')
if __name__=='__main__': main()
