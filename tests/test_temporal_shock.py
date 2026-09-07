import numpy as np
from jcp2026.temporal_shock import decode_temporal_hysteresis


def test_grows_only_seeded_persistent_component_and_excludes_wall():
    p=[]
    for _ in range(3):
        a=np.zeros((20,30)); a[8,8:24]=.7; a[8,8]=.99; a[15,20:25]=.7
        p.append(a)
    domain=[np.ones_like(p[0],bool)]*3; geom=[np.zeros_like(p[0],bool)]*3
    geom[0][8,0:3]=True; geom[1][8,0:3]=True; geom[2][8,0:3]=True
    out=decode_temporal_hysteresis(p,domain,geom,closing_iterations=0,minimum_component_pixels=2)
    assert out[1][8,8:24].all()
    assert not out[1][15,20:25].any()
    assert not (out[1]&geom[1]).any()


def test_requires_temporal_persistence_for_low_score_support():
    p=[np.zeros((12,20)) for _ in range(3)]
    for a in p: a[5,3]=.99
    p[1][5,4:15]=.7
    domain=[np.ones((12,20),bool)]*3; geom=[np.zeros((12,20),bool)]*3
    out=decode_temporal_hysteresis(p,domain,geom,closing_iterations=0,minimum_component_pixels=1)
    assert out[1].sum()==1
