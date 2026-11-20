"""One-component controls for the sealed 15-input, 21-action learner."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import portable_runtime as native
from priority_learner import Learner, Replay

VARIANTS = ("no_double", "no_dueling", "no_per")


class DirectQNetwork(nn.Module):
    """Preserve the paired feature stack and predict action values directly."""

    def __init__(self, dueling_network):
        super().__init__()
        self.feature_layers = deepcopy(dueling_network.feature_layers)
        self.action_head = nn.Sequential(nn.Linear(128, 119), nn.ReLU(), nn.Linear(119, 21))
        self.action_head.apply(dueling_network._init_weights)

    def forward(self, observation):
        return self.action_head(self.feature_layers(observation))


class UniformReplay(Replay):
    """Use the same ring and RNG draw count with unit importance weights."""

    def draw(self, rng, batch=32):
        indices = (rng.random(batch) * self.count).astype(np.int64)
        return indices, np.ones(batch, dtype=np.float32)

    def update(self, indices, residuals):
        pass


class TargetMaxLearner(Learner):
    """Use the target network for both selecting and valuing the next action."""

    def bootstrap(self, next_observation, reward, terminal):
        return reward + .99 * self.target(next_observation).max(1).values * (~terminal)

    def update(self, teacher, stage, imitation_weight):
        replay = self.replay
        indices, weights = replay.draw(self.rng)
        observation = torch.from_numpy(replay.obs[indices])
        next_observation = torch.from_numpy(replay.next_obs[indices])
        action = torch.from_numpy(replay.actions[indices]).view(-1, 1)
        reward = torch.from_numpy(replay.rewards[indices])
        terminal = torch.from_numpy(replay.done[indices])
        values = self.q(observation)
        with torch.no_grad():
            target = self.bootstrap(next_observation, reward, terminal)
        residual = target - values.gather(1, action).squeeze(1)
        td_loss = (torch.from_numpy(weights) * residual.square()).mean()
        imitation_loss = torch.zeros(())
        label = np.full(len(indices), -1, dtype=np.int8)
        if stage:
            mentor, valid = teacher.scores(
                stage, observation, replay.source[indices], replay.energy[indices]
            )
            if bool(valid.any()):
                mask = torch.from_numpy(valid)
                student = values[mask] - values[mask].mean(-1, keepdim=True)
                ideal = mentor[mask]
                imitation_loss = (
                    F.smooth_l1_loss(student, ideal)
                    + .15 * F.kl_div(
                        F.log_softmax(student / .22, dim=-1),
                        F.softmax(ideal / .22, dim=-1),
                        reduction="batchmean",
                    )
                )
                label[valid] = ideal.argmax(dim=-1).numpy().astype(np.int8)

        self.optimizer.zero_grad()
        (td_loss + imitation_weight * imitation_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.)
        self.optimizer.step()
        with torch.no_grad():
            updated_residual = self.bootstrap(next_observation, reward, terminal)
            updated_residual -= self.q(observation).gather(1, action).squeeze(1)
        replay.update(indices, updated_residual.numpy())
        if self.updates % 1000 == 0:
            self.target.load_state_dict(self.q.state_dict())
        self.updates += 1
        return indices, weights, float(td_loss.item()), float(imitation_loss.item()), label


def build_agent(config, variant):
    if variant not in VARIANTS:
        raise ValueError(f"unknown ablation: {variant}")
    agent = native.make_agent(config)
    if variant == "no_dueling":
        agent.q_network = DirectQNetwork(agent.q_network).to(agent.device)
        agent.target_network = deepcopy(agent.q_network)
        agent.optimizer = torch.optim.Adam(agent.q_network.parameters(), lr=agent.learning_rate)
    return agent


def build_learner(config, variant, rng):
    agent = build_agent(config, variant)
    replay = UniformReplay() if variant == "no_per" else Replay()
    learner_type = TargetMaxLearner if variant == "no_double" else Learner
    return agent, learner_type(agent, replay, rng)

