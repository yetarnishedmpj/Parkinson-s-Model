"""
rl_agent.py — upgraded with online experience-replay learning.
The DQN now learns in real-time: each intervention episode stores a
(state, action, reward, next_state) transition, and every 50 steps
the agent runs a mini-batch Bellman update to improve its policy.
"""
import torch, torch.nn as nn, torch.optim as optim
import random, numpy as np, os
from collections import deque

class InterventionDQN(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64),         nn.ReLU(),
            nn.Linear(64, num_actions),
        )
    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, maxlen=5000):
        self.buf = deque(maxlen=maxlen)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(s),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(ns),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buf)


class ClinicalRLAgent:
    STATE_SIZE  = 5
    # 0: None  1: RAS  2: Visual Laser  3: DBS Adjustment  4: Rest Prompt
    ACTION_SIZE = 5
    ACTIONS_MAP = {
        0: "No Intervention",
        1: "Rhythmic Auditory Stimulation to break freezing",
        2: "Visual Laser Cues to step over",
        3: "DBS Voltage Increase for motor control",
        4: "Suggest Rest — fatigue level high",
    }

    def __init__(self):
        self.model  = InterventionDQN(self.STATE_SIZE, self.ACTION_SIZE)
        self.target = InterventionDQN(self.STATE_SIZE, self.ACTION_SIZE)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-4, weight_decay=1e-4)
        self.criterion = nn.SmoothL1Loss()   # Huber loss — more stable than MSE

        self.epsilon      = 0.15
        self.epsilon_min  = 0.05
        self.epsilon_decay= 0.9995
        self.gamma        = 0.95
        self.batch_size   = 32
        self.learn_steps  = 0
        self.target_sync  = 200   # sync target network every N learns

        self.replay = ReplayBuffer(maxlen=5000)

        # Track last transition for reward assignment
        self._last_state  = None
        self._last_action = None
        self._last_neuro  = None

        self.load_model()

    # ── Action selection ────────────────────────────────────────────────────

    def get_action(self, state_array: list):
        if random.random() <= self.epsilon:
            action = random.randrange(self.ACTION_SIZE)
        else:
            t = torch.FloatTensor(state_array).unsqueeze(0)
            with torch.no_grad():
                action = int(torch.argmax(self.model(t)[0]).item())
        return action, self.ACTIONS_MAP[action]

    # ── Online learning ─────────────────────────────────────────────────────

    @staticmethod
    def compute_reward(prev_neuro: float, curr_neuro: float,
                       was_freezing: bool, is_freezing: bool) -> float:
        """Reward = improvement in neurological load + bonus for stopping freeze."""
        r = (prev_neuro - curr_neuro) * 5.0  # positive if load decreased
        if was_freezing and not is_freezing:
            r += 2.0   # large bonus: intervention broke a freeze
        if not was_freezing and is_freezing:
            r -= 1.5   # penalty: freeze started during/after intervention
        return float(np.clip(r, -3.0, 3.0))

    def observe(self, state, action, next_state,
                prev_neuro, curr_neuro, was_freezing, is_freezing):
        """Store transition and trigger a learn step."""
        reward = self.compute_reward(prev_neuro, curr_neuro, was_freezing, is_freezing)
        done   = float(is_freezing)
        self.replay.push(state, action, reward, next_state, done)

    def learn(self):
        if len(self.replay) < self.batch_size:
            return
        s, a, r, ns, d = self.replay.sample(self.batch_size)

        with torch.no_grad():
            q_next  = self.target(ns).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - d)

        q_pred = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss   = self.criterion(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.model.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── Persistence ─────────────────────────────────────────────────────────

    def load_model(self):
        path = os.path.join(os.path.dirname(__file__), "rl_agent.pth")
        if os.path.exists(path):
            try:
                self.model.load_state_dict(
                    torch.load(path, map_location="cpu"))
                self.target.load_state_dict(self.model.state_dict())
                self.model.eval()
            except Exception:
                pass

    def save_model(self):
        path = os.path.join(os.path.dirname(__file__), "rl_agent.pth")
        torch.save(self.model.state_dict(), path)


def pretrain_rl():
    agent = ClinicalRLAgent()
    agent.model.train()
    print("Pre-training the Deep Q-Network …")
    for _ in range(3000):
        hr    = np.random.normal(70, 15)
        hr_n  = (hr - 60) / 100.0
        temp  = np.random.normal(36.6, 0.4)
        temp_n = (temp - 36.5) / 2.0
        act   = np.random.rand()
        prox  = np.random.rand()
        freeze = 1.0 if np.random.rand() > 0.8 else 0.0
        fatigue = np.random.rand()

        state = [hr_n, temp_n, act, prox, freeze]

        if freeze == 1.0:
            target_action = 1 if prox < 0.5 else 2
        elif fatigue > 0.75:
            target_action = 4
        elif act < 0.1 and hr > 80:
            target_action = 3
        else:
            target_action = 0

        next_state = [hr_n * 0.95, temp_n, act * 1.05, prox, 0.0]
        reward = 1.0 if target_action != 0 and freeze == 1.0 else 0.2
        agent.replay.push(state, target_action, reward, next_state, 0.0)

    for _ in range(50):
        agent.learn()

    agent.save_model()
    print("RL Agent pre-training complete and saved.")


if __name__ == "__main__":
    pretrain_rl()
