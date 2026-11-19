"""Array-backed native physics and explicit migration objectives.

This is a new experiment module. It does not modify any sealed native source.
Native mode reproduces the correctness-repaired archived environment. Migration
mode changes only a fixed command grid, observation normalization, and the
declared reward. The optional quadratic_inventory reward adds explicit inventory
feedback to its scalar target, with a fixed384-step horizon. It supplies no
analytic action score, teacher, ramp restriction, terminal energy repair, or
availability override. This is a new branch, not an edit to the frozen main run.
"""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

PACKAGE = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PACKAGE/'data'
FILES = dict(price='price_15min_fullyear.parquet', load='load_15min_fullyear.parquet',
             ev='ev_demand_15min_fullyear.parquet', soh='soh_params_fullyear.json')
EXPECTED = dict(price='35424a55e2e5eefaabe64b184ec82a8cb898595eb8e54dfeda8bf52d8f0fb838',
    load='e8089a38c28a742927b58765205093a8f3fd317b751c18065a268148182fe805',
    ev='c84eec7a97bee6515719fe935ebcccf68dcf51c14e339da4ea8f17ccbbbe8778',
    soh='eda1f3a51ebdf83f9938ead5360bc250236e93312a0d4672634af15b303c60f6')

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class NativeData:
    def __init__(self, directory, stats_rows=17520):
        self.directory=Path(directory).resolve()
        self.paths={k:self.directory/v for k,v in FILES.items()}
        self.hashes={k:_sha(p) for k,p in self.paths.items()}
        assert self.hashes==EXPECTED, 'Input identities differ from archived native sources'
        frames={k:pd.read_parquet(self.paths[k]) for k in ['price','load','ev']}
        self.timestamps=frames['price'].index.to_numpy().copy()
        assert all(np.array_equal(v.index.to_numpy(),self.timestamps) for v in frames.values())
        self.price=frames['price'].price.to_numpy(dtype=np.float64,copy=True)
        self.load=frames['load'].load_kw.to_numpy(dtype=np.float64,copy=True)
        self.ev=frames['ev'].ev_demand_kw.to_numpy(dtype=np.float64,copy=True)
        assert len(self.price)==560640 and all(np.isfinite(x).all() for x in [self.price,self.load,self.ev])
        self.soh_params=json.loads(self.paths['soh'].read_text(encoding='utf8'))
        assert stats_rows==17520, 'Normalization uses the fixed native training prefix'
        l=self.load[:stats_rows];p=self.price[:stats_rows]
        self.train_stats=dict(rows=stats_rows,mean_load=float(l.mean()),var_load=float(l.var()),
            std_load=float(l.std()),mean_base_step_bill=float(np.mean(l*.25*p/1000)))
        assert self.train_stats['var_load']>0 and self.train_stats['mean_base_step_bill']>0
        for value in [self.price,self.load,self.ev,self.timestamps]:value.flags.writeable=False

    def manifest(self):
        return dict(rows=len(self.price),first_timestamp=str(pd.Timestamp(self.timestamps[0])),
            last_timestamp=str(pd.Timestamp(self.timestamps[-1])),
            inputs={k:dict(path=self.paths[k].relative_to(PACKAGE).as_posix(),sha256=self.hashes[k]) for k in FILES},
            training_prefix_rows=17520,training_prefix_unavailable=int(np.sum(self.ev[:17520]<=.1)),
            normalization=dict(self.train_stats),scope='Native archived files, unchanged and SHA-bound')

def load_data(data_config=None):
    config=data_config or {}
    if isinstance(config,(str,Path)):directory=config;stats_rows=17520
    else:
        directory=config.get('directory',config.get('data_dir',DATA_DIRECTORY))
        stats_rows=int(config.get('stats_rows',17520))
    return NativeData(directory,stats_rows)

class MigrationEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,data,config=None):
        super().__init__()
        assert isinstance(data,NativeData)
        self.data=data;self.price=data.price;self.load=data.load;self.ev=data.ev;self.timestamps=data.timestamps
        supplied=copy.deepcopy(config or {})
        self.mode=supplied.get('mode','migration')
        assert self.mode in ['native','migration']
        native=self.mode=='native'
        defaults=dict(mode=self.mode,episode_length=17520,battery_capacity=75.,initial_soc=.5,
            time_step=.25,action_grid=[float(x) for x in np.arange(-10,11)*(5. if native else .3)],
            observation_mode='native' if native else 'normalized15',
            reward_mode='native' if native else 'quadratic_smooth',smooth_weight=1.,economic_weight=0.,
            soc_weight=0.,soh_weight=0.)
        defaults.update(supplied)
        defaults['action_grid']=[float(x) for x in defaults['action_grid']]
        self.config=defaults
        self.action_grid=np.asarray(defaults['action_grid'],np.float64)
        assert len(self.action_grid)>=3 and np.all(np.diff(self.action_grid)>0)
        assert np.any(self.action_grid==0), 'A fixed zero command is required'
        self.zero_action=int(np.flatnonzero(self.action_grid==0)[0])
        if native:
            assert np.array_equal(self.action_grid,np.arange(-10,11)*5.)
            assert defaults['observation_mode']=='native' and defaults['reward_mode']=='native'
        else:assert np.max(abs(self.action_grid))<=3.+1e-12
        assert defaults['battery_capacity']==75. and defaults['initial_soc']==.5 and defaults['time_step']==.25
        assert defaults['observation_mode'] in ['native','normalized15']
        assert defaults['reward_mode'] in ['native','quadratic_smooth','quadratic_inventory']
        self.battery_capacity=75.;self.initial_soc=.5;self.dt=.25;self.soh_params=copy.deepcopy(data.soh_params)
        self.max_power=float(np.max(abs(self.action_grid)))
        self.max_steps=min(int(defaults['episode_length']),len(self.price)-5)
        assert 1<=self.max_steps<=17520
        self.stats=data.train_stats
        self.action_space=spaces.Discrete(len(self.action_grid))
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(15,),dtype=np.float32)
        self.config_digest=hashlib.sha256(json.dumps(self.config,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self._reset_state()

    def _reset_state(self):
        self.current_step=0;self.current_soc=self.initial_soc;self.current_soh=1.
        self.total_cost=0.;self.total_degradation=0.;self.last_action=0.;self.last_command_power=0.
        self.action_history=[self.zero_action]*3;self.action_changes=[];self.action_change_sum=0.;self.done=False

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if options:assert not any(v for v in options.values()), 'Native resets always start source row 0'
        self._reset_state()
        return self._get_observation(),{}

    def _get_observation(self):
        t=self.current_step
        if t>=len(self.price):return np.zeros(15,np.float32)
        prices=[self.price[t]/100.]+[self.price[t+j]/100. if t+j<len(self.price) else 0. for j in range(1,6)]
        current=self.load[t];future=self.load[t+1] if t+1<len(self.load) else None
        if self.config['observation_mode']=='native':
            loads=[current/1000.,future/1000. if future is not None else 0.]
            progress=float(t)/1000.
        else:
            loads=[(current-self.stats['mean_load'])/self.stats['std_load'],
                   (future-self.stats['mean_load'])/self.stats['std_load'] if future is not None else 0.]
            progress=float(t)/17520.
        history=np.asarray(self.action_history[-3:],np.float64)/(len(self.action_grid)-1)
        return np.concatenate([np.asarray(prices),np.asarray(loads),
            np.asarray([self.current_soc,self.current_soh,t%96/96.,progress]),history]).astype(np.float32)

    def _degradation(self,power,old,new):
        params=self.soh_params
        if 'calendar_aging' in params:
            calendar=params['calendar_aging']*self.dt;cycle=params.get('cycle_aging',.00005)
            depth=params.get('depth_factor',1.2);temp=params.get('temperature_factor',1.)
        else:
            calendar=params.get('a',1e-6)*self.dt;cycle=params.get('b',2e-6)
            depth=params.get('c',5e-7)*1000;temp=1.
        cycle_deg=cycle*abs(new-old)*depth*temp if abs(power)>.1 else 0.
        return calendar+cycle_deg

    def step(self,action):
        if self.done:raise RuntimeError('Episode has ended; call reset')
        action=int(action);assert 0<=action<len(self.action_grid)
        t=self.current_step;command=float(self.action_grid[action]);command_change=abs(command-self.last_command_power)
        self.last_command_power=command;old_soc=float(self.current_soc);old_soh=float(self.current_soh)
        available=bool(self.ev[t]>.1)
        power=float(np.clip(command,-old_soc*self.battery_capacity/self.dt,
                            (1-old_soc)*self.battery_capacity/self.dt)) if available else 0.
        change=abs(power-self.last_action);self.last_action=power;self.action_changes.append(change)
        self.action_change_sum+=change
        self.action_history.append(action);self.action_history=self.action_history[-3:]
        energy_delta=power*self.dt
        new_soc=float(np.clip(old_soc+energy_delta/self.battery_capacity,0.,1.))
        degradation=self._degradation(power,old_soc,new_soc)
        self.current_soh-=degradation;self.total_degradation+=degradation
        price=self.price[t];load=self.load[t];cost=power*self.dt*price/1000
        self.total_cost+=cost
        rolling_max=np.max(self.price[max(0,t-672):t+1]);lam=max(price/(rolling_max+1e-9),0.)
        if self.config['reward_mode']=='native':
            economic=lam*(-cost/100.)
            smooth=-abs(load+power-load*.1)/10.
            soh_penalty=-(1.-self.current_soh)*5.;soc_penalty=-abs(old_soc-.5)*2.
            reward=economic+smooth+soh_penalty+soc_penalty
            target=None
        else:
            target=self.stats['mean_load']
            if self.config['reward_mode']=='quadratic_inventory':
                target+=(37.5-old_soc*self.battery_capacity)/(self.dt*384.)
            smooth=-float(self.config['smooth_weight'])*(load+power-target)**2/self.stats['var_load']
            economic=-float(self.config['economic_weight'])*cost/self.stats['mean_base_step_bill']
            soc_penalty=-float(self.config['soc_weight'])*abs(old_soc-.5)
            soh_penalty=-float(self.config['soh_weight'])*(1.-self.current_soh)
            reward=economic+smooth+soh_penalty+soc_penalty
        self.current_soc=new_soc;self.current_step+=1
        self.done=self.current_step>=self.max_steps or self.current_soh<.7
        info=dict(source_index=t,timestamp=str(pd.Timestamp(self.timestamps[t])),load_kw=float(load),price=float(price),
            available=available,ev_available=available,ev_demand_kw=float(self.ev[t]),
            command_power_kw=command,power_kw=power,commanded_power=command,executed_power=power,power_output=power,
            soc_before=old_soc,soc_after=new_soc,soc=new_soc,soh_before=old_soh,soh_after=self.current_soh,soh=self.current_soh,
            energy_before_kwh=old_soc*75.,energy_after_kwh=new_soc*75.,energy_delta_kwh=energy_delta,
            energy_closure_error_kwh=(new_soc-old_soc)*75.-energy_delta,
            step_cost=float(cost),total_cost=float(self.total_cost),degradation=degradation,
            full_net_cost=float((load+power)*self.dt*price/1000),zero_storage_cost=float(load*self.dt*price/1000),
            grid_full_kw=float(load+power),grid_native_kw=float(.9*load+power),
            action_change=change,command_change=command_change,
            current_rosc=(float(np.mean(self.action_changes)) if self.mode=='native' else self.action_change_sum/len(self.action_changes)),
            reward_smooth=float(smooth),reward_economic=float(economic),reward_soc_penalty=float(soc_penalty),
            reward_soh_penalty=float(soh_penalty),reward_target_kw=target,price_weight=float(lam))
        return self._get_observation(),float(reward),bool(self.done),False,info

    def snapshot(self):
        keys=['current_step','current_soc','current_soh','total_cost','total_degradation','last_action',
              'last_command_power','action_history','action_changes','action_change_sum','done']
        result={k:copy.deepcopy(getattr(self,k)) for k in keys}
        result.update(schema='migration_env_state_v1',config_digest=self.config_digest,
            rng=copy.deepcopy(self.np_random.bit_generator.state),
            action_rng=copy.deepcopy(self.action_space.np_random.bit_generator.state))
        return result

    def restore(self,state):
        assert state['schema']=='migration_env_state_v1' and state['config_digest']==self.config_digest
        for key,value in state.items():
            if key not in ['schema','config_digest','rng','action_rng']:setattr(self,key,copy.deepcopy(value))
        self.np_random.bit_generator.state=copy.deepcopy(state['rng'])
        self.action_space.np_random.bit_generator.state=copy.deepcopy(state['action_rng'])
        assert 0<=self.current_step<=self.max_steps and 0<=self.current_soc<=1
        return self._get_observation()

    state=snapshot
    restore_state=restore
