"""Interchange interpolation loops to expose independent time samples.

Each output retains the original weight recurrence and ascending tap-sum
order. A small tile permits SIMD across independent samples without a dense
tap matrix, reduced precision or a coefficient lookup approximation.
"""
import numpy as np
from numba import njit
from ..delay import _apply as scalar_apply

TILE=8


@njit(cache=True,fastmath=False)
def apply_tiled(data,centers,fractions,scales,side_coefficients):
    batch,size=data.shape;query_size=centers.size;half=side_coefficients.size
    weights=np.empty((2*half,TILE));side=np.empty(TILE)
    first=np.empty(TILE,dtype=np.int64);accumulator=np.empty(TILE)
    output=np.empty((batch,query_size));tiles=query_size//TILE
    for tile in range(tiles):
        start=tile*TILE;interior=True
        for lane in range(TILE):
            i=start+lane;fraction=fractions[i]
            side[lane]=fraction*(1.-fraction)
            first[lane]=centers[i]-(half-1)
            if first[lane]<0 or first[lane]+2*half>size:interior=False
        for j in range(1,half):
            for lane in range(TILE):
                i=start+lane;fraction=fractions[i];scale=scales[i]
                side[lane]*=side_coefficients[j]
                weights[half-1-j,lane]=side[lane]/(j+fraction)*scale
                weights[half+j,lane]=side[lane]/(j+1.-fraction)*scale
        for lane in range(TILE):
            i=start+lane;fraction=fractions[i];scale=scales[i]
            weights[half-1,lane]=(1.-fraction)*scale
            weights[half,lane]=fraction*scale
        for b in range(batch):
            for lane in range(TILE):accumulator[lane]=0.
            if interior:
                for k in range(2*half):
                    for lane in range(TILE):
                        accumulator[lane]+=weights[k,lane]*data[b,first[lane]+k]
            else:
                for k in range(2*half):
                    for lane in range(TILE):
                        index=first[lane]+k
                        if 0<=index<size:accumulator[lane]+=weights[k,lane]*data[b,index]
            for lane in range(TILE):output[b,start+lane]=accumulator[lane]
    remaining=tiles*TILE
    if remaining<query_size:
        tail=scalar_apply(data,centers[remaining:],fractions[remaining:],scales[remaining:],side_coefficients)
        output[:,remaining:]=tail
    return output
