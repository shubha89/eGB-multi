"""Compact eccentricity figure from saved scalar overlaps; no waveform runs.

The plotted fraction is derived from rho_template / rho_optimal = overlap.
Every datum and source family is retained; lines are only guides to the eye.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

PALETTE=['#0072B2','#009E73','#E69F00','#B05588','#56B4E9','#D55E00']


def make_figure(rows, figure_dir):
    figure_dir=Path(figure_dir)
    figure_dir.mkdir(parents=True,exist_ok=True)
    paper=figure_dir.parent
    frequencies=sorted({float(r['f0_hz']) for r in rows})
    assert len(rows)==22 and len(frequencies)==6
    assert sum(bool(r.get('current_broad_search_match')) for r in rows)==14
    points=[]
    for r in rows:
        match=float(r['current_orientation_profiled_match'])
        broad=r.get('current_broad_search_match')
        y=100*(1-match)
        assert abs(y/100-(1-match))<1e-15
        points.append(dict(id=r['id'],f0_mHz=float(r['f0_hz'])*1000,
            eccentricity=float(r['eccentricity']),circular_overlap=match,
            circular_snr_loss_percent=y,
            broader_overlap=float(broad) if broad else None,
            broader_snr_loss_percent=100*(1-float(broad)) if broad else None))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,
        'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,
        'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':250}):
        fig,ax=plt.subplots(figsize=(7.05,3.55))
        fig.subplots_adjust(left=.10,right=.985,bottom=.16,top=.88)
        zoom=ax.inset_axes([.055,.49,.42,.455])
        handles=[]
        for frequency,color in zip(frequencies,PALETTE):
            group=sorted([p for p in points if abs(p['f0_mHz']-frequency*1000)<1e-12],key=lambda p:p['eccentricity'])
            xx=[p['eccentricity'] for p in group]
            yy=[p['circular_snr_loss_percent'] for p in group]
            for target in [ax,zoom]:
                target.plot(xx,yy,'o-',color=color,ms=3.6 if target is ax else 3.1,
                    lw=1.15,markeredgecolor='white',markeredgewidth=.35,zorder=3)
                broad=[p for p in group if p['broader_overlap'] is not None]
                target.scatter([p['eccentricity'] for p in broad],
                    [p['broader_snr_loss_percent'] for p in broad],
                    marker='D',s=26 if target is ax else 21,facecolors='white',
                    edgecolors=color,linewidths=.9,zorder=4)
            handles.append(Line2D([],[],color=color,lw=1.8,label=f'{frequency*1000:g}'))
            for p in group:
                if p['broader_snr_loss_percent'] is not None and p['circular_snr_loss_percent']-p['broader_snr_loss_percent']>.5:
                    ax.annotate('',xy=(p['eccentricity'],p['broader_snr_loss_percent']+.8),
                        xytext=(p['eccentricity'],p['circular_snr_loss_percent']-1),
                        arrowprops={'arrowstyle':'->','lw':1.1,'color':color},zorder=2)
        ax.set(xlim=(-.005,.32),ylim=(-1,69),xlabel=r'Initial eccentricity $e_0$',
            ylabel='Unrecovered SNR [%]')
        ax.xaxis.set_major_locator(MultipleLocator(.05))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(axis='y',color='#dedede',lw=.55,zorder=0)
        ax.tick_params(direction='in')
        zoom.set(xlim=(-.003,.108),ylim=(-.5,21),xticks=[0,.05,.1],yticks=[0,10,20])
        zoom.set_xticklabels(['0','0.05','0.10'])
        zoom.tick_params(labelsize=7,direction='in',pad=2)
        zoom.grid(axis='y',color='#e4e4e4',lw=.45,zorder=0)
        zoom.set_title('Low eccentricity',fontsize=8,pad=3)
        for side in zoom.spines.values():side.set_visible(True);side.set_color('#777777');side.set_linewidth(.65)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,1.01),
            ncol=6,frameon=False,title=r'Source families: $f_0$ [mHz]',title_fontsize=8,
            handlelength=1.4,columnspacing=1.5)
        symbol_handles=[Line2D([],[],marker='o',linestyle='none',color='#454545',ms=4,label='Circular templates'),
            Line2D([],[],marker='D',linestyle='none',markerfacecolor='white',markeredgecolor='#454545',ms=4.5,label='Broader circular search')]
        ax.legend(handles=symbol_handles,loc='lower right',frameon=False,
            handletextpad=.55,borderaxespad=.6,fontsize=8)
        for extension in ['pdf','png','svg']:
            fig.savefig(figure_dir/f'restricted_eccentricity_effects.{extension}',bbox_inches='tight')
        plt.close(fig)
    return points
