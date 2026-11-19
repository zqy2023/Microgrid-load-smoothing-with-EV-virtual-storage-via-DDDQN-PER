"""Rebuild the fixed selected beta=2 teacher; no new candidate selection."""
import json
from pathlib import Path
import numpy as np
import teacher as api
ROOT=Path(__file__).resolve().parents[1]
def backward_table(load,price,available,beta,path):
    n=len(load);states=np.arange(3001,dtype=np.int32)
    next_index=np.clip(states[None,:]+np.arange(-10,11,dtype=np.int32)[:,None],0,3000)
    actual=(next_index-states[None,:]).astype(np.float64)/10.
    squared=actual*actual
    table=np.lib.format.open_memmap(path,mode='w+',dtype=np.float64,shape=(n+1,3001))
    offsets=np.empty(n+1,dtype=np.float64)
    terminal=-((states*.025-37.5)*4)**2/n
    offsets[n]=float(terminal.min());table[n]=terminal-offsets[n]
    next_value=np.array(table[n],copy=True)
    coefficients=2*(load-load.mean())+beta*price/4000.
    for t in range(n-1,-1,-1):
        if available[t]:
            candidates=next_value[next_index]+squared+coefficients[t]*actual
            row=np.min(candidates,axis=0)
            center=float(row.min())
            next_value=row-center
        else:
            center=0.
        offsets[t]=offsets[t+1]+center
        table[t]=next_value
    table.flush();del table
    return offsets

def main():
    m=json.loads((ROOT/'config/teacher.json').read_text(encoding='utf8'))
    assert api.sha(__file__)==m['builder_sha256']
    assert api.sha(ROOT/'config/teacher_build.json')==m['protocol_sha256']
    data=api.read_data();folder=ROOT/'config'
    table=(folder/m['value_table_file']).resolve();offset=(folder/m['offsets_file']).resolve()
    table.parent.mkdir(parents=True,exist_ok=True)
    if table.exists() or offset.exists():
        assert table.is_file() and offset.is_file(),'Partial cache exists: use a fresh package/cache directory'
    else:
        offsets=backward_table(data['load'],data['price'],data['available'],2,table)
        np.save(offset,offsets,allow_pickle=False)
    assert api.sha(table)==m['value_table_sha256'],'Table differs from the historical selected teacher'
    assert api.sha(offset)==m['offsets_sha256'],'Offsets differ from the historical selected teacher'
    api.Teacher(ROOT/'config/teacher.json')
    print(json.dumps(dict(status='all_pass',table_sha256=api.sha(table),offsets_sha256=api.sha(offset),new_training_steps=0)))

if __name__=='__main__':main()
