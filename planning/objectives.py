import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        if 'pos' in z_obs_pred:
            loss_visual = metric(z_obs_pred["z"], z_obs_tgt["z"]).mean(
                dim=tuple(range(1, z_obs_pred["z"].ndim))
            )
            loss_pos = metric(z_obs_pred["pos"], z_obs_tgt["pos"]).mean(
                dim=tuple(range(1, z_obs_pred["pos"].ndim))
            )
            loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(1, z_obs_pred["proprio"].ndim))
            )
            loss = loss_visual + alpha * loss_proprio + loss_pos
            return loss_visual, loss_pos, loss_proprio * alpha
        else: 
            if 'z' in z_obs_pred: # for object-centric models
                loss_visual = metric(z_obs_pred["z"][:, -1:], z_obs_tgt["z"]).mean(
                    dim=tuple(range(1, z_obs_pred["z"].ndim))
                )
                loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
                    dim=tuple(range(1, z_obs_pred["proprio"].ndim))
                )
            else:
                loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
                    dim=tuple(range(1, z_obs_pred["visual"].ndim))
                )
                loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
                    dim=tuple(range(1, z_obs_pred["proprio"].ndim))
                )
            loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        if 'z' in z_obs_pred:
            loss_visual = metric(z_obs_pred["z"], z_obs_tgt["z"]).mean(
                dim=tuple(range(2, z_obs_pred["z"].ndim))
            )
        else:
            loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
                dim=tuple(range(2, z_obs_pred["visual"].ndim))
            )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError
