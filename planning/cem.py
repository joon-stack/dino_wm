import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device

from PIL import Image
import matplotlib.pyplot as plt

class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_0 = self.wm.encode_obs(trans_obs_0)
        
    
        if 'overlay' in z_obs_0:
            self.plot_clusters(obs_0['visual'].astype(int), z_obs_0['overlay'], 'start')
            self.plot_clusters(obs_g['visual'].astype(int), z_obs_g['overlay'], 'goal')

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
            losses_visual = []
            losses_pos = []
            losses_proprio = []
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                with torch.no_grad():
                    i_z_obses, i_zs = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )                
                if 'z' in i_z_obses:
                    match_objects = self.wm.match_objects
                    cur_obs = torch.cat([cur_z_obs_g['z'], cur_z_obs_g['pos']], dim=-1)
                    i_obs = torch.cat([i_z_obses['z'][:, -1:], i_z_obses['pos'][:, -1:]], dim=-1)
                    cur_obs_dct = {
                        'z': cur_obs[:, :, :, :self.wm.encoder.emb_dim], 
                        'pos': cur_obs[:, :, :, self.wm.encoder.emb_dim:],
                        'proprio': cur_z_obs_g['proprio'],
                    }
                    i_obs_dct = {
                        'z': i_obs[:, :, :, :self.wm.encoder.emb_dim],
                        'pos': i_obs[:, :, :, self.wm.encoder.emb_dim:],
                        'proprio': i_z_obses['proprio'][:, -1:],
                    }

                    loss_visual, loss_pos, loss_proprio = self.objective_fn(cur_obs_dct, i_obs_dct)
                    loss = loss_visual + loss_pos + loss_proprio

                
                else:
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

                try:
                    losses_visual.append(loss_visual[topk_idx[0]].item())
                    losses_pos.append(loss_pos[topk_idx[0]].item())
                    losses_proprio.append(loss_proprio[topk_idx[0]].item())
                except:
                    pass

            try:
                self.wandb_run.log(
                    {
                        f"{self.logging_prefix}/loss": np.mean(losses), 
                        f"{self.logging_prefix}/loss_visual": np.mean(losses_visual),
                        f"{self.logging_prefix}/loss_pos": np.mean(losses_pos),
                        f"{self.logging_prefix}/loss_proprio": np.mean(losses_proprio),
                        "step": i + 1,
                    }
                )

                print(f"INFO: {self.logging_prefix}/loss: {np.mean(losses):.3f}, {self.logging_prefix}/loss_visual: {np.mean(losses_visual):.3f}, {self.logging_prefix}/loss_pos: {np.mean(losses_pos):.3f}, {self.logging_prefix}/loss_proprio: {np.mean(losses_proprio):.3f}")
            except:
                self.wandb_run.log(
                    {
                        f"{self.logging_prefix}/loss": np.mean(losses),
                        "step": i + 1,
                    }
                )
                print(f"INFO: {self.logging_prefix}/loss: {np.mean(losses):.3f}")

            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid

    def plot_clusters(self, frame, overlay, filename):
        overlay = overlay.detach().cpu().numpy()
        for b in range(frame.shape[0]):
            for t in range(frame.shape[1]):
        
                plt.figure(figsize=(8, 8))
                plt.imshow(frame[b][t])
                plt.axis('off')
                plt.imshow(overlay[b][t], alpha=0.8)
                plt.show()
                plt.savefig(f"{filename}_{b}_{t}.png")
                plt.close()
