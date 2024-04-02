import numpy as np
import torch as T
from model import ContinousActorNetwork, ContinuousCriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims,
                 n_actions, agent_idx, agent_name,
                 gamma=0.99, alpha=3e-4, T=2048,
                 gae_lambda=0.95, policy_clip=0.2,
                 batch_size=64, n_epochs=10,
                 n_procs=8, chkpt_dir=None,
                 scenario=None):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = 1e-3
        self.agent_idx = agent_idx
        self.agent_name = agent_name
        self.n_procs = n_procs

        self.actor = ContinousActorNetwork(n_actions, actor_dims, alpha,
                                          chkpt_dir=chkpt_dir,
                                          scenario=scenario)
        self.critic = ContinuousCriticNetwork(critic_dims, alpha,
                                              chkpt_dir=chkpt_dir,
                                              scenario=scenario)
        self.n_actions = n_actions

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        with T.no_grad():
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)

            dist = self.actor(state)
            action = dist.sample()
            probs = dist.log_prob(action)

        return action.cpu().numpy(), probs.cpu().numpy()

    def calc_adv_and_returns(self, memories):
        states, new_states, r, dones = memories
        with T.no_grad():
            values = self.critic(states).squeeze()
            values_ = self.critic(new_states).squeeze()
            deltas =r[:,:, self.agent_idx] + self.gamma * values_ * (1 - dones) - values

            adv = [0]

            for step in reversed(range(deltas.shape[0])):
                advantage = deltas[step] + self.gamma * self.gae_lambda * adv[-1] * np.array(dones[step])
                adv.append(advantage)

            adv.reverse()
            adv = np.array(adv[:-1])
            adv = T.tensor(adv, device=self.critic.device).unsqueeze(2)

