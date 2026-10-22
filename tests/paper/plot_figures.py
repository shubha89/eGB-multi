"""Replay all six paper figures from the restored compact plotting data.

No waveform evaluation, noise realization, integration or search is launched.
Source-illustration recalculation is a separate explicit command. Timings shown
in Figure 6 are the recorded measurements, never fresh timing measurements.
"""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,io,json,shutil,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_eccentricity_effects import make_figure

HERE=Path(__file__).resolve().parent
SOURCE_STYLE={'font.size':10,'axes.labelsize':10,'legend.fontsize':8,
    'savefig.dpi':240,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}
COMPARISON_STYLE={'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':250}
NAMES=['source_harmonics','periastron_envelope','pm_rates','annual_covariance',
    'restricted_eccentricity_effects','source_validation_and_timing']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(fig,out,name):
    for ext in ('pdf','png'):fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)
def read_csv(path):return list(csv.DictReader(Path(path).open()))

def fig1(data,out):
    a=np.load(data/'source_harmonics.npz',allow_pickle=False)
    fig,axes=plt.subplots(3,2,figsize=(7.2,5.7),layout='constrained')
    for row,e in enumerate((0.,.3,.6)):
        p=a[f'harmonic_power_{row}']
        axes[row,0].plot(a['time_hours'],a[f'hplus_normalized_{row}'],lw=.8)
        axes[row,0].set_ylabel(rf'$h_+$, $e={e:g}$')
        axes[row,1].bar(a['harmonic_f_over_f0'],p,width=.15,color='#287ca5')
        axes[row,1].set(yscale='log',ylim=(1e-8,2),xlim=(.25,6.3))
        axes[row,1].axhline(.1,c='.5',ls='--',lw=.8)
        axes[row,1].set_ylabel('relative power')
        axes[row,1].text(.98,.95,rf'$N_{{\geq10\%}}={sum(p>=.1)}$',transform=axes[row,1].transAxes,ha='right',va='top')
    axes[0,0].set_title('Fixed Newtonian source');axes[0,1].set_title('Harmonic power')
    axes[2,0].set_xlabel('time [hours]');axes[2,1].set_xlabel(r'$f/f_0$')
    save(fig,out,NAMES[0])

def annotate_half_period(path,period):
    from pypdf import PdfReader,PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    page=PdfReader(path).pages[0];width,height=float(page.mediabox.width),float(page.mediabox.height)
    rects=Counter(tuple(float(v) for v in args) for args,op in page.get_contents().operations if op==b're' and len(args)==4)
    boxes=sorted([v for v in rects if v[2]>.75*width and .2*height<v[3]<.5*height],key=lambda v:v[1])
    if len(boxes)!=2:raise ValueError('Expected the two conservative-envelope axes')
    x0,y0,w,h=boxes[0];marker_x=x0+w*(period/2)/356
    buf=io.BytesIO();c=canvas.Canvas(buf,pagesize=(width,height),invariant=1)
    c.setStrokeColorRGB(.30,.30,.30);c.setLineWidth(.7);c.setDash(3,2)
    for x,y,bw,bh in boxes:c.line(marker_x,y,marker_x,y+bh)
    c.setDash();lx=marker_x+5;ly=y0+h-11
    tw=pdfmetrics.stringWidth('T','Helvetica-Oblique',9);sw=pdfmetrics.stringWidth('\u03d6','Symbol',6.5)
    suffix=f'/2 = {period/2:.1f} d';total=tw+sw+pdfmetrics.stringWidth(suffix,'Helvetica',9)
    c.setFillColorRGB(1,1,1);c.rect(lx-2,ly-4,total+4,14,fill=1,stroke=0);c.setFillColorRGB(.1,.1,.1)
    c.setFont('Helvetica-Oblique',9);c.drawString(lx,ly,'T')
    c.setFont('Symbol',6.5);c.drawString(lx+tw,ly-2,'\u03d6')
    c.setFont('Helvetica',9);c.drawString(lx+tw+sw,ly,suffix);c.save();buf.seek(0)
    page.merge_page(PdfReader(buf).pages[0]);writer=PdfWriter();writer.add_page(page)
    writer.add_metadata({'/Title':'Conservative periastron envelope with apsidal half-period marker',
        '/Subject':'Saved waveform illustration; annotation-only restoration'})
    with Path(path).open('wb') as stream:writer.write(stream)

def fig2(data,out):
    a=np.load(data/'periastron_envelope.npz',allow_pickle=False);period=float(a['full_apsidal_period_days'])
    fig,ax=plt.subplots(2,1,figsize=(7.2,4.8),layout='constrained')
    for prefix,color,label in [('newtonian','#cf6726','Newtonian'),('onepn','#287ca5','1PN orbit')]:
        ax[0].fill_between(a['days'],a[prefix+'_min'],a[prefix+'_max'],color=color,alpha=.45,label=label,lw=0)
    ax[0].set(xlim=(0,356),ylabel=r'normalized $h_+$ range');ax[0].legend(ncol=2)
    ax[1].plot(a['rms_days'],a['rms_ratio'],color='#13876a',lw=1)
    ax[1].axhline(1,c='.5',ls='--',lw=.7)
    ax[1].set(xlim=(0,356),xlabel='time [days]',ylabel='RMS envelope ratio')
    ax[1].text(.03,.94,rf'$T_\varpi={period:.1f}$ d',transform=ax[1].transAxes,va='top')
    pdf=out/'periastron_envelope.pdf';fig.savefig(pdf,bbox_inches='tight')
    annotate_half_period(pdf,period)
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm','-r','240','-singlefile','-png',str(pdf),str(pdf.with_suffix(''))],check=True,capture_output=True)
    else:
        for axis in ax:axis.axvline(period/2,c='.3',ls='--',lw=.7)
        ax[1].annotate(rf'$T_\varpi/2={period/2:.1f}$ d',xy=(period/2,.94),
            xycoords=('data','axes fraction'),xytext=(5,0),textcoords='offset points',va='top',fontsize=9)
        fig.savefig(pdf.with_suffix('.png'),bbox_inches='tight')
    plt.close(fig)

def fig3(data,out):
    rows=read_csv(data/'pm_rate_data.csv');fig,ax=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
    for mass,color in ((.5,'#287ca5'),(1.4,'#cf6726')):
        for e,ls in ((0.,'-'),(.1,'--'),(.3,'-.'),(.5,':')):
            group=[r for r in rows if float(r['m1_solar'])==mass and float(r['e'])==e]
            f=[float(r['f0_hz']) for r in group]
            label=rf'{mass:g}+{mass:g} $M_\odot$, $e={e:g}$'
            ax[0].loglog(f,[float(r['phase_estimate_rad']) for r in group],color=color,ls=ls,lw=1.2,label=label)
            if e:ax[1].loglog(f,[float(r['fractional_e_estimate']) for r in group],color=color,ls=ls,lw=1.2)
    for axis in ax:axis.set_xlabel(r'$f_0$ [Hz]');axis.grid(alpha=.2)
    ax[0].set_ylabel(r'$\dot n_0 T^2$ [rad]');ax[1].set_ylabel(r'$|\dot e_0|T/e_0$')
    ax[0].legend(fontsize=6.7,ncol=2);save(fig,out,NAMES[2])

def fig4(data,out):
    a=np.load(data/'annual_covariance_plot.npz',allow_pickle=False);q=a['frequency_hz'];c=a['covariance']
    fig,ax=plt.subplots(2,1,figsize=(3.6,4.1),layout='constrained',sharex=True)
    for j,label in enumerate(('A','E')):ax[0].loglog(q,c[:,j,j].real,label=label,lw=1)
    ax[0].set_ylabel(r'$S_{II}$ [Hz$^{-1}$]');ax[0].legend()
    coherence=abs(c[:,0,1])**2/(c[:,0,0].real*c[:,1,1].real)
    ax[1].loglog(q,coherence,lw=1,color='#13876a')
    ax[1].set(xlim=(1e-5,.08),xlabel='frequency [Hz]',ylabel=r'$|S_{AE}|^2/(S_{AA}S_{EE})$')
    for axis in ax:axis.grid(alpha=.2)
    save(fig,out,NAMES[3])

def fig6(data,out):
    rows=read_csv(data/'pyefpe_source_validation.csv');x=np.arange(len(rows))
    labels=[rf"{float(r['f0_mHz']):g} mHz"+'\n'+rf"$e_0={float(r['eccentricity']):g}$" for r in rows]
    blue,orange='#236a99','#b85b16';fig,axes=plt.subplots(1,2,figsize=(7.3,3.0),layout='constrained')
    axes[0].semilogy(x,[float(r['common_relative_l2']) for r in rows],'o-',color=blue,label='Shared Newtonian/Peters physics',ms=4)
    axes[0].semilogy(x,[float(r['higher_PN_relative_l2']) for r in rows],'s--',color=orange,label='Different PN source models',ms=4)
    axes[0].set(ylabel=r'$\epsilon_h$',title='Two-polarization relative residual');axes[0].legend(loc='upper left',fontsize=7)
    width=.34
    axes[1].bar(x-width/2,[float(r['egb_source_s']) for r in rows],width,label='eGB NumPy/Peters',color=blue)
    axes[1].bar(x+width/2,[float(r['pyefpe_source_s']) for r in rows],width,label='pyEFPE harmonic TD adapter',color=orange)
    axes[1].set(yscale='log',ylabel='wall time [s]',title='One year of source polarizations');axes[1].legend(loc='upper left',fontsize=7)
    for axis in axes:
        axis.set_xticks(x,labels);axis.tick_params(labelsize=8);axis.grid(axis='y',alpha=.2,linewidth=.5);axis.set_axisbelow(True)
    for ext in ('pdf','png','svg'):fig.savefig(out/f'{NAMES[5]}.{ext}',bbox_inches='tight')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=HERE/'data')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();data=args.data.resolve();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty or new figure-output directory')
    manifest=json.loads((HERE/'plot_data_manifest.json').read_text())
    for name,expected in manifest['input_sha256'].items():
        if digest(data/name)!=expected:raise ValueError('Changed plotting input: '+name)
    out.mkdir(parents=True,exist_ok=True)
    for number,fn in enumerate([fig1,fig2,fig3,fig4],1):
        with plt.rc_context(SOURCE_STYLE):fn(data,out)
        print('Replayed Figure',number,flush=True)
    with plt.rc_context(COMPARISON_STYLE):
        make_figure(read_csv(data/'representative_sources_updated.csv'),out)
        fig6(data,out)
    result=dict(status='passed_six_figure_replay',figures=6,waveform_evaluations=0,
        searches_launched=0,timing_measurements=0,input_sha256=manifest['input_sha256'],
        output_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()},
        scope='Figures replay saved plotting data; Figure 6 times are recorded observations, not new measurements.')
    (out/'figure_replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
