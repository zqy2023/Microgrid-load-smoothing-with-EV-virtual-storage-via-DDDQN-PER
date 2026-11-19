"""Read-only full-future offline teacher for the fixed archived 21-action task.

Scores are centered negative DP action costs divided by the load variance:
higher is better, best action is zero. This supplies supervised action values,
not a causal policy or evidence of independent reinforcement-learning discovery.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=PACKAGE/'data'
EXPECTED={
    'load_15min_fullyear.parquet':'e8089a38c28a742927b58765205093a8f3fd317b751c18065a268148182fe805',
    'price_15min_fullyear.parquet':'35424a55e2e5eefaabe64b184ec82a8cb898595eb8e54dfeda8bf52d8f0fb838',
    'ev_demand_15min_fullyear.parquet':'c84eec7a97bee6515719fe935ebcccf68dcf51c14e339da4ea8f17ccbbbe8778',
    'soh_params_fullyear.json':'eda1f3a51ebdf83f9938ead5360bc250236e93312a0d4672634af15b303c60f6',
}
N=17520
CAPACITY=75.
INITIAL=37.5
DT=.25
UNIT=.025
ACTION_GRID=np.arange(-10,11,dtype=np.float64)/10.
ACTION_ORDER=np.asarray(sorted(range(21),key=lambda j:(abs(float(ACTION_GRID[j])),j)),dtype=np.int64)
ZERO_INDEX=10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_data():
    for name,expected in EXPECTED.items():assert sha(DATA/name)==expected,name
    frames={key:pd.read_parquet(DATA/name) for key,name in [
        ('load','load_15min_fullyear.parquet'),('price','price_15min_fullyear.parquet'),('ev','ev_demand_15min_fullyear.parquet')]}
    assert all(len(frame)==560640 for frame in frames.values())
    assert all(np.array_equal(frame.index,frames['load'].index) for frame in frames.values())
    return dict(load=frames['load'].load_kw.to_numpy(dtype=np.float64)[:N],
        price=frames['price'].price.to_numpy(dtype=np.float64)[:N],
        available=frames['ev'].ev_demand_kw.to_numpy(dtype=np.float64)[:N]>.1,
        timestamps=frames['load'].index[:N].astype(str).to_numpy(dtype=str))


class Teacher:
    def __init__(self,manifest_path=None,verify=True,data=None):
        manifest_path=Path(manifest_path or PACKAGE/'config/teacher.json').resolve()
        self.manifest=json.loads(manifest_path.read_text(encoding='utf8'))
        folder=manifest_path.parent
        self.table_path=(folder/self.manifest['value_table_file']).resolve()
        self.offset_path=(folder/self.manifest['offsets_file']).resolve()
        if verify:
            assert self.manifest['teacher_source_sha256']==sha(__file__)
            assert self.manifest['value_table_sha256']==sha(self.table_path)
            assert self.manifest['offsets_sha256']==sha(self.offset_path)
            assert self.manifest['input_sha256']==EXPECTED
        self.values=np.load(self.table_path,mmap_mode='r',allow_pickle=False)
        self.offsets=np.load(self.offset_path,mmap_mode='r',allow_pickle=False)
        assert self.values.shape==(N+1,3001) and self.values.dtype==np.float64
        assert self.offsets.shape==(N+1,)
        self.data=read_data() if data is None else data
        self.load=self.data['load'];self.price=self.data['price'];self.available=self.data['available']
        self.mean_load=float(self.load.mean());self.var_load=float(self.load.var())
        self.beta=float(self.manifest['beta'])

    def action_costs(self,t_array,e_array):
        """Unscaled stage+next centered V; smaller is better. Scalar -> [21]."""
        ts=np.asarray(t_array,dtype=np.int64);es=np.asarray(e_array,dtype=np.float64)
        ts,es=np.broadcast_arrays(ts,es);scalar=ts.ndim==0
        assert ts.ndim<=1
        ts=ts.reshape(-1);es=es.reshape(-1)
        assert np.all((ts>=0)&(ts<N)) and np.isfinite(es).all()
        assert np.all((es>=-1e-10)&(es<=CAPACITY+1e-10))
        es=np.clip(es,0,CAPACITY)
        command=np.broadcast_to(ACTION_GRID,(len(ts),21))
        actual=np.minimum(np.maximum(command,-es[:,None]/DT),(CAPACITY-es[:,None])/DT)
        actual=np.where(self.available[ts,None],actual,0.)
        next_energy=np.clip(es[:,None]+DT*actual,0,CAPACITY)
        position=next_energy/UNIT
        nearest=np.rint(position)
        position=np.where(abs(position-nearest)<=1e-7,nearest,position)
        lo=np.floor(position).astype(np.int64);hi=np.minimum(lo+1,3000)
        alpha=position-lo
        future=(1-alpha)*self.values[ts[:,None]+1,lo]+alpha*self.values[ts[:,None]+1,hi]
        terminal=ts==N-1
        if np.any(terminal):
            future[terminal]=-((next_energy[terminal]-INITIAL)*4)**2/N-self.offsets[N]
        coefficient=2*(self.load[ts]-self.mean_load)+self.beta*self.price[ts]/4000.
        costs=actual*actual+coefficient[:,None]*actual+future
        return costs[0] if scalar else costs

    def action_values(self,t_array,e_array):
        """[B,21] centered scores; higher better, max0; scalar -> [21]."""
        costs=self.action_costs(t_array,e_array)
        return -(costs-np.min(costs,axis=-1,keepdims=True))/self.var_load

    def action(self,t_array,e_array):
        costs=self.action_costs(t_array,e_array)
        result=ACTION_ORDER[np.argmin(costs[...,ACTION_ORDER],axis=-1)]
        available=self.available[np.asarray(t_array,dtype=np.int64)]
        result=np.where(available,result,ZERO_INDEX)
        return int(result) if np.ndim(result)==0 else result.astype(np.int64)
