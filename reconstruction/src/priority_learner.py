"""Array/tree implementation of the archived Double Dueling DQN update.

The tree replaces linear-time sampling. In the transfer methods the loss has
a separately logged action-advantage teaching term. The TD target, optimizer,
replay size, cadence and network architecture follow the preceding B run.
"""
import numpy as np
import torch
from torch.nn import functional as F


class Replay:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.leaves = 1 << (capacity - 1).bit_length()
        self.weights = np.zeros(2*self.leaves, dtype=np.float64)
        self.obs = np.empty((capacity, 15), dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.done = np.empty(capacity, dtype=bool)
        self.source = np.empty(capacity, dtype=np.int32)
        self.energy = np.empty(capacity, dtype=np.float64)
        self.count = 0
        self.written = 0
        self.maximum = 1.
        self.beta = .4

    def add(self, obs, action, reward, nxt, done, source, energy):
        i = self.written % self.capacity
        self.obs[i] = obs
        self.next_obs[i] = nxt
        self.actions[i] = action
        self.rewards[i] = reward
        self.done[i] = done
        self.source[i] = source
        self.energy[i] = energy
        node = self.leaves+i
        self.weights[node] = np.float32(self.maximum)
        node //= 2
        while node:
            self.weights[node] = self.weights[2*node] + self.weights[2*node+1]
            node //= 2
        self.count = min(self.capacity, self.count+1)
        self.written += 1

    def draw(self, rng, batch=32):
        mass = rng.random(batch) * self.weights[1]
        nodes = np.ones(batch, dtype=np.int64)
        for _ in range(self.leaves.bit_length()-1):
            nodes *= 2
            left = self.weights[nodes]
            right = mass >= left
            nodes += right
            mass -= np.where(right, left, 0.)
        probabilities = self.weights[nodes]/self.weights[1]
        indices = nodes-self.leaves
        weights = (self.count*probabilities)**(-self.beta)
        weights = (weights/weights.max()).astype(np.float32)
        self.beta = min(1., self.beta+.001)
        assert indices.min() >= 0 and indices.max() < self.count
        return indices, weights

    def update(self, indices, residuals):
        priorities = np.asarray((np.abs(residuals).astype(np.float64)+1e-5)**.6, dtype=np.float32)
        self.maximum = max(self.maximum, float(priorities.max()))
        nodes = self.leaves + indices
        # Repeated indices have the same residual. Order is retained as well, so
        # the last write wins, just as in the reference buffer.
        self.weights[nodes] = priorities
        for _ in range(self.leaves.bit_length()-1):
            nodes = nodes//2
            self.weights[nodes] = self.weights[2*nodes]+self.weights[2*nodes+1]


class Learner:
    def __init__(self, agent, replay, rng):
        self.q = agent.q_network
        self.target = agent.target_network
        self.optimizer = agent.optimizer
        self.replay = replay
        self.rng = rng
        self.updates = 0

    def action(self, obs, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.randint(21))
        with torch.no_grad():
            return int(self.q(torch.from_numpy(obs).view(1,15)).argmax().item())

    def update(self, teacher, stage, imitation_weight):
        replay = self.replay
        indices, weights = replay.draw(self.rng)
        x = torch.from_numpy(replay.obs[indices])
        nxt = torch.from_numpy(replay.next_obs[indices])
        act = torch.from_numpy(replay.actions[indices]).view(-1,1)
        r = torch.from_numpy(replay.rewards[indices])
        terminal = torch.from_numpy(replay.done[indices])
        values = self.q(x)
        with torch.no_grad():
            next_actions = self.q(nxt).argmax(1,keepdim=True)
            td_target = r + .99*self.target(nxt).gather(1,next_actions).squeeze(1)*(~terminal)
        residual = td_target-values.gather(1,act).squeeze(1)
        td_loss = (torch.from_numpy(weights)*residual.square()).mean()
        imitation_loss = torch.zeros(())
        label = np.full(len(indices), -1, dtype=np.int8)
        if stage:
            mentor, valid = teacher.scores(
                stage, x, replay.source[indices], replay.energy[indices],
            )
            if bool(valid.any()):
                mask = torch.from_numpy(valid)
                student = values[mask] - values[mask].mean(-1,keepdim=True)
                ideal = mentor[mask]
                # A value fit gives the shared network feedback over all 21 slots.
                # The soft ranking term concentrates on actions near the optimum.
                imitation_loss = (
                    F.smooth_l1_loss(student, ideal)
                    + .15*F.kl_div(F.log_softmax(student/.22,dim=-1),
                                   F.softmax(ideal/.22,dim=-1), reduction="batchmean")
                )
                label[valid] = ideal.argmax(dim=-1).numpy().astype(np.int8)
        loss = td_loss + imitation_weight*imitation_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(),10.)
        self.optimizer.step()
        with torch.no_grad():
            new_best = self.q(nxt).argmax(1,keepdim=True)
            new_td = r + .99*self.target(nxt).gather(1,new_best).squeeze(1)*(~terminal)
            new_td -= self.q(x).gather(1,act).squeeze(1)
        replay.update(indices,new_td.numpy())
        if self.updates % 1000 == 0:
            self.target.load_state_dict(self.q.state_dict())
        self.updates += 1
        return indices, weights, float(td_loss.item()), float(imitation_loss.item()), label


def verify_sampler():
    # Compare sampled slots and weights to a direct cumulative distribution,
    # including the overwrite phase. Use the same random variates.
    rng = np.random.RandomState(842)
    oracle = np.random.RandomState(842)
    buffer = Replay(97)
    obs = np.zeros(15,dtype=np.float32)
    for t in range(320):
        buffer.add(obs,0,0.,obs,False,0,37.5)
        before = buffer.weights[buffer.leaves:buffer.leaves+buffer.count].copy()
        cdf = np.cumsum(before)
        target = np.searchsorted(cdf,oracle.random(32)*cdf[-1],side='right')
        beta = buffer.beta
        ix,w = buffer.draw(rng)
        expected=(buffer.count*before[target]/cdf[-1])**(-beta)
        assert np.array_equal(target,ix)
        assert np.allclose(w,expected/expected.max(),rtol=2e-7,atol=2e-7)
        residual = np.arange(32,dtype=np.float32)/32+1+t*.01
        residual=np.asarray([residual[np.flatnonzero(ix==i)[-1]] for i in ix])
        buffer.update(ix,residual)
        expected = before.copy()
        expected[ix] = ((np.abs(residual).astype(np.float64)+1e-5)**.6).astype(np.float32)
        assert np.array_equal(buffer.weights[buffer.leaves:buffer.leaves+buffer.count],expected)
        assert np.isclose(buffer.weights[1],expected.sum(dtype=np.float64),rtol=1e-15)
    return dict(status="all_pass", cases=320, includes_ring_overwrites=True)

