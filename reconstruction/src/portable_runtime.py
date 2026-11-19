"""Portable bindings; original network/configuration helpers are preserved."""
import copy, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
from native_dddqn_per_sb3 import DDDQN_PER_Agent, PrioritizedReplayBuffer
from migration_env import MigrationEnv, load_data
ROOT=Path(__file__).resolve().parents[1]
OLD_CONFIG=ROOT/'config/full.json'
NATIVE_PARAMETERS = dict(state_size=15, learning_rate=3e-4, gamma=.99,
    epsilon=1., epsilon_min=.01, epsilon_decay=.995, buffer_size=200000,
    batch_size=32, update_frequency=4, target_update_frequency=1000)

TRAINING_DEFAULTS = dict(algorithm='full', dropout=.2, epsilon_schedule='native',
                         linear_steps=120000, linear_final_epsilon=.05)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

def normalized_config(config, data):
    assert not (set(config) - {'environment', 'training', 'data', 'description'}), 'Unknown config blocks'
    supplied = config.get('training', {})
    assert not (set(supplied) - set(TRAINING_DEFAULTS)), 'Native hyperparameters are fixed'
    train = dict(TRAINING_DEFAULTS); train.update(supplied)
    assert train['algorithm'] in ['full', 'no_per']
    assert 0 <= float(train['dropout']) < 1
    assert train['epsilon_schedule'] in ['native', 'linear120k']
    assert int(train['linear_steps']) == 120000
    assert 0 <= float(train['linear_final_epsilon']) <= 1
    env = MigrationEnv(data, config.get('environment', {}))
    return dict(environment=env.config, training=train,
                data=copy.deepcopy(config.get('data', {})),
                description=config.get('description', ''))

class UniformReplay(PrioritizedReplayBuffer):
    """Declared no-PER control; native train_step and RNG forwards are retained."""
    def sample(self, batch_size):
        size = len(self.buffer)
        indices = np.random.choice(size, batch_size, replace=True)
        batch = [self.buffer[int(i)] for i in indices]
        weights = np.ones(batch_size, dtype=np.float64)
        self.beta = min(1., self.beta + self.beta_increment)
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        pass

def make_agent(config):
    agent = DDDQN_PER_Agent(action_size=len(config['environment']['action_grid']),
                           device=torch.device('cpu'), **NATIVE_PARAMETERS)
    for network in [agent.q_network, agent.target_network]:
        for module in network.modules():
            if isinstance(module, torch.nn.Dropout): module.p = float(config['training']['dropout'])
        network.train()
    if config['training']['algorithm'] == 'no_per': agent.memory = UniformReplay(200000)
    return agent

def expected_updates(steps): return 0 if steps < 32 else (steps - 32)//4 + 1

def file_binding(path):
    path=Path(path).resolve()
    return dict(path=Path(os.path.relpath(path,ROOT)).as_posix(),sha256=sha(path),bytes=path.stat().st_size)

def resolve(path):
    path=Path(path)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()

def source_bindings():
    files=[Path(__file__),Path(__file__).with_name('migration_env.py'),Path(__file__).with_name('native_dddqn_per_sb3.py')]
    return {p.name:file_binding(p) for p in files}
