"""Rebuild all visualizations from saved results; never invokes any learner."""
from collections import defaultdict,Counter
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io import inside,clean,write_json

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':.18,'savefig.dpi':180,'pdf.fonttype':42,'svg.fonttype':'none'})
COLORS=['#2166ac','#b35806','#4d9221','#762a83','#555555','#1b9e77']

def load_runs(results):
    runs=[]
    for path in sorted(Path(results).rglob('metrics.json')):
        data=json.loads(path.read_text(encoding='utf-8'));data['_folder']=str(path.parent)
        if 'configuration' not in data:
            from .config import read_config
            data['configuration']=read_config(path.parent/'resolved_config.yaml')
        if 'implementation_hash' not in data:
            data['implementation_hash']=json.loads((path.parent/'metadata.json').read_text(encoding='utf-8')).get('implementation_hash')
        stage_path=path.parent/'stage_diagnostics.jsonl'
        data['_stages']=[json.loads(line) for line in stage_path.read_text(encoding='utf-8').splitlines() if line] if stage_path.exists() else []
        runs.append(data)
    return runs

def label(run):
    arch=run['architecture'];width='-'.join(map(str,arch['hidden_widths']))
    cfg=run.get('configuration',{});numeric=cfg.get('oracle',{});recovery=cfg.get('recovery',{})
    suffix=''
    if run['experiment_name']=='numerical_ablation':
        suffix=f" | {numeric.get('backend','?')}, dps={numeric.get('dps','?')}, step={recovery.get('reduced_gradient_step_factor','?')}, tau={cfg.get('final_layer',{}).get('tau_fin','?')}"
    method=run.get('method','Cubature control' if run['kind']=='cubature_control' else 'ASPIRE')
    return f"{method} / {run['experiment_name']} / {run['teacher_family']}\n{run['oracle_mode']} | {run['sampler_mode']} | d={arch['d']}, h={width}, k={arch['k']}"+suffix

def source_caption(runs):
    return f"Source: {len(runs)} saved run artifacts | Curves: run median / IQR; other panels: measured values or counts | failures retained"

def save(fig,path,runs):
    fig.text(.01,.005,source_caption(runs),fontsize=7,color='#555555')
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+ext),bbox_inches='tight')
    plt.close(fig)
    write_json(path.with_suffix('.sources.json'),{'source_runs':[r['run_id'] for r in runs],
        'source_folders':[r['_folder'] for r in runs],'aggregation':'Curves: run median/IQR; heatmaps, spectra and counts: individual observations; descriptive statistics only'})

def parameter_error(run):
    return run.get('single_layer_operator_error') if run['kind']=='single_layer' else run.get('max_weight_operator_error')

def curve(ax,runs,x_key,y_fn,xlabel,ylabel,title):
    groups=defaultdict(list)
    for r in runs:
        x=r.get(x_key);y=y_fn(r)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            # Query counts differ slightly across seeds due to exact cache hits.
            # Aggregate the planned operating point, plot its median actual cost.
            bucket=r.get('angular_directions',r.get('label_budget',r.get('moment_samples',x))) if x_key=='n_real_learning_queries' else x
            groups[label(r)].append((bucket,x,y))
    for index,(name,pairs) in enumerate(groups.items()):
        by_x=defaultdict(list)
        for bucket,x,y in pairs:by_x[bucket].append((x,y))
        buckets=sorted(by_x);xs=[np.median([v[0] for v in by_x[b]]) for b in buckets]
        median=[np.median([v[1] for v in by_x[b]]) for b in buckets]
        lo=[np.quantile([v[1] for v in by_x[b]],.25) for b in buckets];hi=[np.quantile([v[1] for v in by_x[b]],.75) for b in buckets]
        color=COLORS[index%len(COLORS)]
        ax.plot(xs,median,'o-',color=color,label=name,markersize=4)
        ax.fill_between(xs,lo,hi,color=color,alpha=.12)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=title)
    ax.set_yscale('symlog',linthresh=1e-12)
    if groups:ax.legend(fontsize=6,loc='best')

def main_plots(runs,output):
    generated=[]
    parameter=[r for r in runs if r['status']!='invalid' and r['experiment_name'] in ('sample_scaling','single_layer_debug')]
    if parameter:
        fig,axes=plt.subplots(2,2,figsize=(13,8.5))
        strict=[r for r in parameter if r['kind']=='recovery' and r['oracle_mode']=='strict_real']
        single=[r for r in parameter if r['kind']=='single_layer']
        curve(axes[0,0],strict,'moment_samples',lambda r:r['per_layer_errors'][0] if r.get('per_layer_errors') else None,
              'Retained moment samples (points)','First-layer operator error','Strict first layer / all returned first layers')
        curve(axes[0,1],single,'moment_samples',parameter_error,'Retained moment samples (points)','Single-layer operator error','Exact suffix diagnostic / fixed instance family')
        curve(axes[1,0],strict,'n_real_learning_queries',parameter_error,'Median actual real learning queries (calls)','Maximum layer operator error','Full-network error / completed networks only')
        curve(axes[1,1],strict,'n_real_learning_queries',lambda r:r.get('output_l1_error'),'Median actual real learning queries (calls)','Output vector L1 error','Output weights / completed networks only')
        for ax in axes.flat:ax.axhline(.1,ls='--',lw=.8,color='gray')
        save(fig,output/'parameter_recovery',parameter);generated.append('parameter_recovery.png')
    recovered=[r for r in runs if r.get('per_layer_errors')]
    if recovered:
        data=np.full((len(recovered),max(len(r['per_layer_errors']) for r in recovered)),np.nan)
        for i,r in enumerate(recovered):data[i,:len(r['per_layer_errors'])]=r['per_layer_errors']
        fig,ax=plt.subplots(figsize=(9,max(3.3,len(recovered)*.28)))
        im=ax.imshow(data,aspect='auto',cmap='viridis',vmin=0)
        ax.set_xticks(np.arange(data.shape[1]),[f'Layer {i+1}' for i in range(data.shape[1])])
        ax.set_yticks(np.arange(len(recovered)),[f"{r['experiment_name']} / seed {r['seed']} / {r['oracle_mode']}" for r in recovered],fontsize=7)
        ax.set(title='Consistently aligned per-layer operator error',xlabel='Hidden layer',ylabel='Saved run');ax.grid(False)
        fig.colorbar(im,ax=ax,label='Operator norm error')
        save(fig,output/'layerwise_error',recovered);generated.append('layerwise_error.png')
    gaussian=[r for r in runs if r.get('empirical_gaussian_nmse') is not None]
    if gaussian:
        fig,axes=plt.subplots(1,2,figsize=(13,4.7))
        curve(axes[0],gaussian,'n_real_learning_queries',lambda r:r.get('empirical_gaussian_nmse'),'Real learning queries (calls)','Gaussian NMSE','Ordinary Gaussian test / prediction risk')
        curve(axes[1],gaussian,'n_real_learning_queries',lambda r:r.get('angular_gaussian_nmse'),'Real learning queries (calls)','Angular Gaussian NMSE','Analytic radial integration / network predictors')
        save(fig,output/'gaussian_comparison',gaussian);generated.append('gaussian_comparison.png')
    if runs:
        relevant=[r for r in runs if r.get('n_real_learning_queries',0)>0]
        if relevant:
            fig,ax=plt.subplots(figsize=(11,max(3.5,len(relevant)*.28)))
            y=np.arange(len(relevant));left=np.zeros(len(relevant))
            for i,scope in enumerate(('span','membership','moment_gradient','final_hessian','training','validation')):
                values=np.array([r['query_counts'].get('n_real_calls_'+scope,0) for r in relevant])
                if not values.any():continue
                ax.barh(y,values,left=left,label=scope,color=COLORS[i%len(COLORS)]);left+=values
            ax.set_yticks(y,[f"{r['experiment_name']} / {r.get('method','ASPIRE')} / {r['seed']}" for r in relevant],fontsize=7)
            ax.set(xlabel='Real learning queries (calls); evaluation excluded',ylabel='Saved run',title='Actual query ledger by purpose')
            ax.legend(fontsize=7);save(fig,output/'query_costs',relevant);generated.append('query_costs.png')
        counts=Counter(('complete: target met' if r.get('parameter_success') else 'complete: target not met') if r['kind']=='recovery' and r['status']=='complete' else r['status']+': '+(r.get('failure_reason') or r['kind']) for r in runs)
        fig,ax=plt.subplots(figsize=(10,max(3,len(counts)*.4)))
        ax.barh(list(counts),list(counts.values()),color=COLORS[0]);ax.set(xlabel='Runs (count)',ylabel='Outcome',title='All saved outcomes; no failed seeds removed')
        save(fig,output/'run_outcomes',runs);generated.append('run_outcomes.png')
    return generated

def diagnostic_plots(runs,output):
    runs=[r for r in runs if r['status']!='invalid']
    generated=[]
    gap=[r for r in runs if r.get('estimated_population_gap') is not None and parameter_error(r) is not None]
    if gap:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for family in sorted(set(r['teacher_family'] for r in gap)):
            subset=[r for r in gap if r['teacher_family']==family]
            ax.scatter([r['estimated_population_gap'] for r in subset],[parameter_error(r) for r in subset],label=family)
        ax.set(xlabel='Estimated population relative gap (dimensionless)',ylabel='Weight operator error',title='Gap diagnostic / truth-visible reference integration')
        ax.set_xscale('symlog',linthresh=1e-8);ax.legend(fontsize=8)
        save(fig,output/'gap_recovery',gap);generated.append('gap_recovery.png')
    oracle=[r for r in runs if r.get('oracle_accuracy_rows')]
    if oracle:
        rows=[v for r in oracle for v in r['oracle_accuracy_rows']]
        fig,ax=plt.subplots(figsize=(8,4.5))
        for dps in sorted(set(v['dps'] for v in rows)):
            vals=sorted([v for v in rows if v['dps']==dps],key=lambda v:v['degree'])
            ax.plot([v['degree'] for v in vals],[v['relative_error'] for v in vals],'o-',label='float64' if dps==0 else f'{dps} decimal digits')
        ax.set_yscale('log');ax.set(xlabel='Total polynomial degree Q',ylabel='Relative complex-value error',title='Fourier simulation / measured precision error');ax.legend()
        save(fig,output/'oracle_precision',oracle);generated.append('oracle_precision.png')
    prefix=[r for r in runs if r.get('actual_prefix_operator_error') is not None]
    if prefix:
        fig,axes=plt.subplots(1,2,figsize=(11,4.4))
        curve(axes[0],prefix,'actual_prefix_operator_error',lambda r:r.get('suffix_relative_error_median'),'Measured prefix operator error','Median suffix relative error','Prefix perturbation / suffix values')
        curve(axes[1],prefix,'actual_prefix_operator_error',lambda r:r.get('gradient_error_median'),'Measured prefix operator error','Median gradient absolute error','Prefix perturbation / interpolated gradients')
        save(fig,output/'prefix_propagation',prefix);generated.append('prefix_propagation.png')
    sampling=[]
    for r in runs:
        path=Path(r['_folder'])/'artifacts'/'arrays.npz'
        if path.exists():
            with np.load(path) as arrays:
                if 'radial_uniform' in arrays:sampling.append((r,arrays['radial_uniform'].copy(),arrays.get('samples',np.zeros((0,2))).copy()))
    if sampling:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        axes[0].plot([0,1],[0,1],'k--',label='Uniform reference')
        seen=set()
        for r,values,z in sampling:
            name=f"{r['experiment_name']} / m={r['moment_samples']}"+(f" / xi={r['xi']}" if r['experiment_name']=='fuzzy_sampling' else '')
            ordered=np.sort(values);axes[0].plot(ordered,np.arange(1,len(values)+1)/len(values),lw=.8,alpha=.7,label=name if name not in seen else '_nolegend_')
            seen.add(name)
        r,values,z=sampling[0]
        if z.shape[1]>=2:axes[1].scatter(z[:,0],z[:,1],s=3,alpha=.25)
        axes[0].set(xlabel='G(Z)^(n/q)',ylabel='Empirical cumulative probability',title='Radial distribution (does not certify angular mixing)')
        axes[0].legend(fontsize=6)
        axes[1].set(xlabel='Reduced coordinate z1',ylabel='Reduced coordinate z2',title=f"2D projection / {r['experiment_name']}")
        save(fig,output/'sampling_diagnostics',[v[0] for v in sampling]);generated.append('sampling_diagnostics.png')
    return generated

def write_summary(runs,output):
    from .experiments.aggregate import cluster_summary
    write_json(output/'teacher_cluster_summary.json',cluster_summary(runs))
    fields=['experiment_name','run_id','kind','method','seed','teacher_seed','algorithm_seed','status','failure_reason',
        'oracle_mode','sampler_mode','Q','moment_samples','integration_order','angular_directions','parameter_success','max_weight_operator_error','single_layer_operator_error',
        'output_l1_error','empirical_gaussian_nmse','angular_gaussian_nmse','n_real_learning_queries','n_real_evaluation_queries','wall_time_total']
    with (output/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in runs:writer.writerow({field:r.get(field) for field in fields})

def report_html(runs,images,figures,report):
    import os
    records=clean([{k:v for k,v in r.items() if k!='_stages'} for r in runs])
    data=json.dumps(records,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    gallery=''.join(f'<figure><a href="{html.escape(os.path.relpath(figures/name,report).replace(chr(92),chr(47)))}"><img loading="lazy" src="{html.escape(os.path.relpath(figures/name,report).replace(chr(92),chr(47)))}" alt="{html.escape(name)}"></a><figcaption>{html.escape(name)}</figcaption></figure>' for name in images)
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASPIRE Numerical Experiment Report</title><style>
body{font:15px/1.65 system-ui,sans-serif;color:#202932;background:#f8fafc;margin:0}main{max-width:1250px;margin:auto;padding:32px}h1{font-size:28px;margin:0}h2{font-size:21px;margin-top:34px}.note{color:#576674;max-width:950px}.stats{display:flex;gap:32px;padding:20px 0;border-bottom:1px solid #dce2e8}.stat strong{display:block;font-size:24px;color:#2166ac}select{padding:7px;margin:12px 12px 12px 0}table{width:100%;border-collapse:collapse;background:white;font-size:12px}th,td{padding:9px;text-align:left;border-bottom:1px solid #e1e6ea}th{background:#eaf0f5}details{margin:8px 0}pre{white-space:pre-wrap;max-height:400px;overflow:auto;font-size:11px}.gallery{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0;background:white;padding:12px}figure img{width:100%}figcaption{font-size:12px;color:#52606d}button{border:0;background:none;color:#2166ac;cursor:pointer}#detail{background:white;padding:12px}.scroll{overflow:auto}@media(max-width:850px){.gallery{grid-template-columns:1fr}.stats{flex-wrap:wrap}}</style>
<main><h1>ASPIRE Numerical Experiments</h1><p class="note">Generated from saved numerical results. Real-query recovery, truth-based diagnostics and passive baselines are labeled separately. Completed runs need not meet the accuracy target. All results are empirical_only.</p>
<div class="stats" id="stats"></div><h2>Run Records</h2><div id="filters"></div><div class="scroll"><table><thead><tr><th>Experiment / seed</th><th>Mode</th><th>Status</th><th>Parameter Error</th><th>Gaussian NMSE</th><th>Learning Queries</th><th>Time / s</th><th>Details</th></tr></thead><tbody id="rows"></tbody></table></div><div id="detail" hidden></div>
<h2>Scientific Figures</h2><p class="note">Figures use saved observations only. Read accuracy plots together with run status. Each figure has a .sources.json provenance record and PDF/SVG exports.</p><div class="gallery">GALLERY</div>
<p class="note">Rebuild with python run_experiments.py --plots --report. Configurations and outputs are stored in configs/ and results/.</p></main>
<script>const runs=DATA;const $=id=>document.getElementById(id);const esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>typeof x==='number'?(x===0?'0':Math.abs(x)<.01?x.toExponential(2):x.toLocaleString(undefined,{maximumFractionDigits:3})):'—';
const counts=[['Saved Runs',runs.length],['Completed Runs',runs.filter(r=>r.status==='complete').length],['Hit-and-Run Accuracy Passes',runs.filter(r=>r.kind==='recovery'&&r.oracle_mode==='strict_real'&&r.parameter_success).length],['Cubature Control Passes',runs.filter(r=>r.kind==='cubature_control'&&r.parameter_success).length],['Failed / Error',runs.filter(r=>r.status!=='complete').length]];$('stats').innerHTML=counts.map(([n,v])=>`<div class="stat"><strong>${v}</strong>${n}</div>`).join('');
const keys=['kind','oracle_mode','sampler_mode','status'];$('filters').innerHTML=keys.map(k=>`<label>${esc(k)} <select id="${k}"><option value="">All</option>${[...new Set(runs.map(r=>r[k]))].map(v=>`<option>${esc(v)}</option>`).join('')}</select></label>`).join('');
function render(){const selected=runs.filter(r=>keys.every(k=>!$(k).value||r[k]===$(k).value));$('rows').innerHTML=selected.map(r=>`<tr><td>${esc(r.experiment_name)} / ${r.seed}${r.method?' / '+esc(r.method):''}</td><td>${esc(r.oracle_mode)}<br>${esc(r.sampler_mode)}</td><td>${esc(r.status)}${r.failure_reason?'<br>'+esc(r.failure_reason):''}${r.kind==='recovery'&&r.status==='complete'?'<br>'+(r.parameter_success?'Accuracy passed':'Accuracy not reached'):''}</td><td>${fmt(r.max_weight_operator_error??r.single_layer_operator_error)}</td><td>${fmt(r.empirical_gaussian_nmse)}</td><td>${fmt(r.n_real_learning_queries)}</td><td>${fmt(r.wall_time_total)}</td><td><button data-id="${runs.indexOf(r)}">Expand</button></td></tr>`).join('');document.querySelectorAll('button[data-id]').forEach(b=>b.onclick=()=>{const r=runs[+b.dataset.id];$('detail').hidden=false;$('detail').innerHTML='<h3>'+esc(r.run_id)+'</h3><pre>'+esc(JSON.stringify(r,null,2))+'</pre>';});}keys.forEach(k=>$(k).onchange=render);render();</script></html>'''
    (report/'index.html').write_text(document.replace('GALLERY',gallery).replace('DATA',data),encoding='utf-8')

def build_all(results,output,report):
    output=inside(output);report=inside(report);output.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    runs=load_runs(results)
    if not runs:print('No saved results: no placeholder charts generated.');return
    images=main_plots(runs,output)+diagnostic_plots(runs,output)
    write_summary(runs,report);report_html(runs,images,output,report)
    print(f'Built {len(images)} figure groups and offline report: {report / "index.html"}',flush=True)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results');parser.add_argument('--output',default='figures')
    args=parser.parse_args();build_all(inside(args.results),inside(args.output),inside('reports'))

if __name__=='__main__':main()
