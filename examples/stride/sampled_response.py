"""Standalone CPU interface smoke example with a simple synthetic triangle."""
import argparse
import jax
jax.config.update('jax_enable_x64',True)
import numpy as np
from lisa_stride import LINK_LABELS,DetectorResponse,SampledPolarizations

p=argparse.ArgumentParser();p.add_argument('--backend',choices=['jax','numba'],default='jax')
a=p.parse_args()
t=np.arange(1024)*2.5;vertices=np.array([[500.,0.,0.],[508.3,0.,.1],[504.,7.1,-.2]])
sender=[];receiver=[];directions=[];lengths=[]
for label in LINK_LABELS:
 r,e=int(label[0])-1,int(label[1])-1
 d=vertices[r]-vertices[e];L=np.linalg.norm(d)
 sender.append(np.tile(vertices[e],(len(t),1)));receiver.append(np.tile(vertices[r],(len(t),1)))
 directions.append(np.tile(d/L,(len(t),1)));lengths.append(np.full(len(t),L))
g=dict(times=t,n=np.array(directions),length=np.array(lengths),p_sender=np.array(sender),p_receiver=np.array(receiver))
options={'device':'cpu'} if a.backend=='jax' else {}
detector=DetectorResponse.prepare(g,backend=a.backend,**options)
origin=-5000.;dt=.625;q=origin+np.arange(16384)*dt
source=SampledPolarizations.from_arrays(np.cos(2*np.pi*.003*q),.7*np.sin(2*np.pi*.003*q),origin_s=origin,dt_s=dt,backend=a.backend)
k=np.array([0.,0.,-1.]);plus=np.diag([1.,-1.,0.]);cross=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,0.]])
result=detector.sampled_response(source,k,plus,cross)
result=np.asarray(result)
assert result.shape==(3,len(t)) and np.all(np.isfinite(result))
print(a.backend,'XYZ',result.shape,'L2',float(np.linalg.norm(result)))
