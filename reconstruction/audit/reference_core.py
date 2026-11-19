"""Independent physics/15D/named-weight matrix reference for the new single-network study.

Copied unchanged calculation definitions from the separately audited reference
verify_learned.py SHA256 d21e27b92f522cc50bc3a594dc131d76c3c6380f9115112a060ad72b8ff2d2ce.
No experiment modules are imported. The new verifier supplies initial weights.
"""
import argparse

import hashlib

import json

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

N = 17520

EXPECTED_INPUTS = {
    'price': '35424a55e2e5eefaabe64b184ec82a8cb898595eb8e54dfeda8bf52d8f0fb838',
    'load': 'e8089a38c28a742927b58765205093a8f3fd317b751c18065a268148182fe805',
    'ev': 'c84eec7a97bee6515719fe935ebcccf68dcf51c14e339da4ea8f17ccbbbe8778',
    'soh': 'eda1f3a51ebdf83f9938ead5360bc250236e93312a0d4672634af15b303c60f6',
}

EXPECTED_NATIVE = {
    'migration_train.py': '2e14e1c07d51e881b8a44198e5a208847110e29cb52d43a5fcbe32904b0127c1',
    'migration_env.py': '66a97bfeb82e8fb100485bda1fac93dfdb84e8c3184e4be0af122be8f4c65931',
    'native_dddqn_per_sb3.py': '0f0d6db3fa25b17999b2238ce6a2f4b064b7303abe59a7cc064d17eb52194ca3',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def bound(binding):
    path = Path(binding['path']); path = path.resolve() if path.is_absolute() else (Path(__file__).resolve().parents[1]/path).resolve()
    assert path.is_file() and sha(path) == binding['sha256'], str(path)
    if 'bytes' in binding:
        assert path.stat().st_size == binding['bytes']
    return path

def close(actual, expected, atol=1e-9, label='value'):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
    assert error <= atol, (label, error, atol)
    return error

def weights_valid(state):
    layers = {'feature_layers.0': (128, 15), 'feature_layers.3': (128, 128),
              'value_stream.0': (64, 128), 'value_stream.2': (1, 64),
              'advantage_stream.0': (64, 128), 'advantage_stream.2': (21, 64)}
    assert set(state) == {name + '.' + kind for name in layers for kind in ['weight', 'bias']}
    for name, shape in layers.items():
        assert tuple(state[name + '.weight'].shape) == shape
        assert tuple(state[name + '.bias'].shape) == (shape[0],)
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all().item() for v in state.values())
    assert sum(v.numel() for v in state.values()) == 36502

def manual_q(observation, state):
    def linear(value, name):
        return F.linear(value, state[name + '.weight'], state[name + '.bias'])
    with torch.no_grad():
        x = torch.as_tensor(observation, dtype=torch.float32)
        x = torch.relu(linear(x, 'feature_layers.0'))
        x = torch.relu(linear(x, 'feature_layers.3'))
        value = linear(torch.relu(linear(x, 'value_stream.0')), 'value_stream.2')
        advantage = linear(torch.relu(linear(x, 'advantage_stream.0')), 'advantage_stream.2')
        return (value + advantage - advantage.mean(1, keepdim=True)).numpy()

def verify_q(observations, saved, actions, state):
    batch_max = 0.
    rechecks = 0
    for first in range(0, N, 256):
        last = min(N, first + 256)
        q = manual_q(observations[first:last], state)
        batch_max = max(batch_max, float(np.max(abs(q - saved[first:last]))))
        flags = np.flatnonzero((~np.isclose(q, saved[first:last], rtol=2e-6, atol=2e-5)).any(1)
                               | (q.argmax(1) != actions[first:last]))
        for off in flags:
            row = first + int(off)
            single = manual_q(observations[row:row+1], state)[0]
            rechecks += 1
            assert np.allclose(single, saved[row], rtol=2e-7, atol=2e-6), ('Q', row)
            assert int(single.argmax()) == int(actions[row]), ('argmax', row)
    return dict(rows=N, q_mismatch_rows=0, action_mismatch_rows=0,
                maximum_batch256_difference=batch_max, batch1_rechecks=rechecks,
                batch_tolerance=dict(rtol=2e-6, atol=2e-5),
                batch1_tolerance=dict(rtol=2e-7, atol=2e-6),
                method='Named FP32 linear/ReLU/dueling equations, no model.forward')

def verify_trace(path, data, grid, recorded_metrics, neutral, state):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    expected_keys = {'actions', 'observations', 'q_values', 'power_kw', 'command_kw',
                     'applied_command_kw', 'energy_kwh', 'load_kw', 'price', 'available',
                     'net_kw', 'step_bill'}
    assert set(arrays) == expected_keys
    for key, value in arrays.items():
        shape = (N+1,) if key == 'energy_kwh' else (N, 15) if key == 'observations' else (N, 21) if key == 'q_values' else (N,)
        assert value.shape == shape and np.isfinite(value).all(), (key, value.shape)
    actions = arrays['actions']
    assert np.issubdtype(actions.dtype, np.integer) and np.all((actions >= 0) & (actions < 21))
    assert arrays['observations'].dtype == np.float32 and arrays['q_values'].dtype == np.float32
    assert np.array_equal(arrays['q_values'].argmax(1), actions), 'Recorded Q/action consistency'
    price, load, ev, soh_params = data
    available = ev[:N] > .1
    assert np.array_equal(arrays['load_kw'], load[:N])
    assert np.array_equal(arrays['price'], price[:N])
    assert np.array_equal(arrays['available'], available)
    close(arrays['command_kw'], grid[actions], atol=0., label='command')
    close(arrays['net_kw'], load[:N] + arrays['power_kw'], atol=0., label='net power')
    close(arrays['step_bill'], (load[:N] + arrays['power_kw']) * price[:N] / 4000,
          atol=0., label='step bill')
    soc, soh = .5, 1.
    energies = [37.5]
    reconstructed_obs = np.empty((N, 15), np.float32)
    actual = np.empty(N)
    applied = np.empty(N)
    history = [int(np.flatnonzero(grid == 0)[0])] * 3
    remaining = np.r_[np.cumsum(available[::-1])[::-1], 0]
    mean, std = load[:N].mean(), load[:N].std()
    maximum = float(np.max(abs(grid)))
    for t in range(N):
        reconstructed_obs[t] = np.r_[price[t:t+6] / 100.,
            (load[t:t+2] - mean) / std, soc, soh, (t % 96) / 96., t / N,
            np.asarray(history[-3:]) / 20.]
        selected = float(grid[actions[t]])
        if neutral and available[t]:
            gap = (37.5 - soc * 75.) / .25
            remaining_power = remaining[t+1] * maximum
            selected = float(np.clip(selected, gap - remaining_power, gap + remaining_power))
        applied[t] = selected
        power = float(np.clip(selected, -soc * 75. / .25, (1. - soc) * 75. / .25)) if available[t] else 0.
        actual[t] = power
        next_soc = float(np.clip(soc + power * .25 / 75., 0., 1.))
        if 'calendar_aging' in soh_params:
            calendar = soh_params['calendar_aging'] * .25
            cycle = soh_params.get('cycle_aging', .00005)
            depth = soh_params.get('depth_factor', 1.2)
            temp = soh_params.get('temperature_factor', 1.)
        else:
            calendar = soh_params.get('a', 1e-6) * .25
            cycle = soh_params.get('b', 2e-6)
            depth = soh_params.get('c', 5e-7) * 1000
            temp = 1.
        soh -= calendar + (cycle * abs(next_soc-soc) * depth * temp if abs(power) > .1 else 0.)
        soc = next_soc
        energies.append(soc * 75.)
        history.append(int(actions[t]))
        assert soh >= .7, ('Unexpected SoH terminal', t)
    errors = dict(
        observation=close(arrays['observations'], reconstructed_obs, atol=2e-7, label='15D observations'),
        power=close(arrays['power_kw'], actual, atol=1e-9, label='actual power'),
        applied_command=close(arrays['applied_command_kw'], applied, atol=1e-9, label='adjusted command'),
        energy=close(arrays['energy_kwh'], energies, atol=1e-9, label='energy'),
        conservation=close(np.diff(arrays['energy_kwh']), arrays['power_kw']*.25, atol=1e-9, label='conservation'))
    assert max(abs(actual)) <= maximum + 1e-9 and min(energies) >= -1e-9 and max(energies) <= 75+1e-9
    assert np.all(actual[~available] == 0.)
    if neutral:
        assert abs(energies[-1] - energies[0]) < 1e-8, 'Equal endpoint energy'
    p = arrays['power_kw']; energy = arrays['energy_kwh']; net = load[:N] + p
    baseline = float(np.dot(load[:N], price[:N]) / 4000)
    metric = dict(
        pooled_pct=float(100*(1-net.var()/load[:N].var())),
        daily_pct=float(100*(1-net[:17472].reshape(182,96).var(1).mean()/load[:17472].reshape(182,96).var(1).mean())),
        saving_pct=float(-100*np.dot(p,price[:N])/4000/baseline),
        baseline_bill=baseline, bill=float(np.dot(net,price[:N])/4000),
        final_energy_kwh=float(energy[-1]), energy_change_kwh=float(energy[-1]-energy[0]),
        energy_min_kwh=float(energy.min()), energy_max_kwh=float(energy.max()),
        power_min_kw=float(p.min()), power_max_kw=float(p.max()),
        unavailable_slots=int((~available).sum()),
        viability_adjusted_steps=int(np.count_nonzero(abs(arrays['command_kw']-arrays['applied_command_kw'])>1e-9)),
        availability_evaluation='archived EV demand > 0.1')
    assert set(recorded_metrics) == set(metric)
    for key, value in metric.items():
        if isinstance(value, str):
            assert value == recorded_metrics[key]
        else:
            assert abs(value-recorded_metrics[key]) < 1e-9, (key, value, recorded_metrics[key])
    inference = verify_q(arrays['observations'], arrays['q_values'], actions, state) if state is not None else None
    off_grid = np.min(abs(arrays['applied_command_kw'][:,None]-grid[None,:]), axis=1) > 1e-9
    return dict(status='all_pass', trace_path=str(path), trace_sha256=sha(path), rows=N,
                neutral_wrapper=neutral, independent_weights_available_for_this_trace=state is not None,
                independent_Q=inference, maximum_errors=errors, metrics=metric,
                adjusted_commands_outside_original_grid=int(off_grid.sum()),
                scope='Native physical equations and source-derived 15D observations; complete days exclude 48-row tail')
