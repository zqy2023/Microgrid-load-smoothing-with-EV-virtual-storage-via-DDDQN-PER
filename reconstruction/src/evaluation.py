"""The original evaluation equations with package-local imports."""
from pathlib import Path
import numpy as np
import torch
from migration_env import MigrationEnv,load_data
import portable_runtime as native
ROOT=Path(__file__).resolve().parents[1]
N=17520
VARIANTS={'C':dict(inventory_hours=144,price_shift_kw=.025,cubic_kw=0.)}
class RetunedEnv(MigrationEnv):
    def __init__(self, data, config, variant, neutral=False):
        super().__init__(data, config)
        self.variant = variant
        self.neutral = neutral
        self.available = data.ev[:N] > 0.1
        self.remaining_availability = np.r_[
            np.cumsum(self.available[::-1])[::-1], 0
        ]
        self.center_price = float(self.price[:N].mean())
        self.std_price = float(self.price[:N].std())

    def step(self, action):
        t = self.current_step
        if self.neutral:
            # Limit execution only when a command would make it impossible to
            # get back to the starting energy in the remaining available slots.
            # The training and the primary evaluation keep this disabled.
            gap = (37.5 - self.current_soc * 75) / self.dt
            remaining = self.remaining_availability[t + 1] * self.max_power
            selected = float(self.action_grid[action])
            adjusted = float(np.clip(selected, gap - remaining, gap + remaining))
            if not self.available[t]:
                adjusted = selected
            self.action_grid[action] = adjusted
            try:
                obs, _, done, truncated, info = super().step(action)
            finally:
                self.action_grid[action] = selected
            info["command_power_kw"] = selected
            info["viability_adjusted_command_kw"] = adjusted
        else:
            obs, _, done, truncated, info = super().step(action)
        delta = info["load_kw"] - self.stats["mean_load"]
        before = info["energy_before_kwh"]
        feedback = (37.5 - before) / self.variant["inventory_hours"]
        feedback += self.variant["cubic_kw"] * ((37.5 - before) / 37.5) ** 3
        power = info["power_kw"]
        standard_price = (info["price"] - self.center_price) / self.std_price
        smooth = (delta ** 2 - (delta + power - feedback) ** 2) / self.stats["var_load"]
        economy = -2 * self.variant["price_shift_kw"] * standard_price * power / self.stats["var_load"]
        return obs, smooth + economy, done, truncated, info

def metrics(load, price, p, energy):
    d = (N // 96) * 96
    net = load + p
    base = float(np.dot(load, price) / 4000)
    return dict(
        pooled_pct=float(100 * (1 - np.var(net) / np.var(load))),
        daily_pct=float(100 * (1 - net[:d].reshape(-1, 96).var(axis=1).mean()
                                 / load[:d].reshape(-1, 96).var(axis=1).mean())),
        saving_pct=float(-100 * np.dot(p, price) / 4000 / base),
        baseline_bill=base,
        bill=float(np.dot(net, price) / 4000),
        final_energy_kwh=float(energy[-1]),
        energy_change_kwh=float(energy[-1] - energy[0]),
        energy_min_kwh=float(energy.min()),
        energy_max_kwh=float(energy.max()),
        power_min_kw=float(p.min()), power_max_kw=float(p.max()),
    )

def evaluate(env, agent, path=None, zero=False):
    obs, _ = env.reset()
    actions = np.empty(N, np.int16)
    observations = np.empty((N, 15), np.float32)
    q_values = np.empty((N, 21), np.float32)
    energy = np.empty(N + 1)
    power = np.empty(N)
    command = np.empty(N)
    executed_command = np.empty(N)
    energy[0] = 37.5
    agent.q_network.eval()
    with torch.inference_mode():
        for t in range(N):
            observations[t] = obs
            q_values[t] = agent.q_network(torch.from_numpy(obs).reshape(1, -1)).numpy()[0]
            if zero:
                action = env.zero_action
            else:
                action = int(q_values[t].argmax())
            actions[t] = action
            obs, _, done, truncated, info = env.step(action)
            if truncated or done != (t == N - 1):
                raise RuntimeError(f"Unexpected episode boundary at {t}")
            energy[t + 1] = info["energy_after_kwh"]
            power[t] = info["power_kw"]
            command[t] = info["command_power_kw"]
            executed_command[t] = info.get("viability_adjusted_command_kw", command[t])
    agent.q_network.train()
    load = env.load[:N]
    price = env.price[:N]
    available = env.ev[:N] > 0.1
    if (np.max(np.abs(np.diff(energy) - .25 * power)) > 1e-9
            or np.max(np.abs(power)) > env.max_power + 1e-9
            or np.any(power[~available] != 0) or energy.min() < -1e-9 or energy.max() > 75+1e-9):
        raise RuntimeError("Trace breaks the archived physical constraints")
    if env.neutral and abs(energy[-1] - energy[0]) > 1e-8:
        raise RuntimeError("Equal-energy evaluation failed")
    result = metrics(load, price, power, energy)
    result.update(unavailable_slots=int((~available).sum()),
                  viability_adjusted_steps=int(np.count_nonzero(np.abs(command-executed_command)>1e-9)),
                  availability_evaluation="archived EV demand > 0.1")
    if path is not None:
        np.savez_compressed(path, actions=actions, observations=observations, q_values=q_values,
                            power_kw=power, command_kw=command, applied_command_kw=executed_command,
                            energy_kwh=energy, load_kw=load, price=price, available=available,
                            net_kw=load+power, step_bill=(load+power)*price/4000)
    return result
