import torch
import decord
import numpy as np
from pathlib import Path
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced, TrajSlicerDataset
decord.bridge.set_bridge("torch")

import numpy as np

# precomputed dataset stats
ACTION_MEAN = torch.tensor([0.0006, 0.0015])
ACTION_STD = torch.tensor([0.4395, 0.4684])
STATE_MEAN = torch.tensor([0.7518, 0.9239, -3.9702e-05, 3.1550e-04])
STATE_STD = torch.tensor([1.0964, 1.2390, 1.3819, 1.5407])

class WallDataset(TrajDataset):
    def __init__(
        self,
        data_path: str = "data/wall_single",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
    ):  
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        print("Loading wall dataset from self.data_path")
        states = torch.load(self.data_path / "states.pth")
        self.states = states
        self.proprios = self.states.clone()
        self.actions = torch.load(self.data_path / "actions.pth")
        self.actions = self.actions / action_scale
        self.door_locations = torch.load(self.data_path / "door_locations.pth")
        self.wall_locations = torch.load(self.data_path / "wall_locations.pth")

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)
            print(f"Loaded {n} rollouts")

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.proprios = self.proprios[:n]
        self.door_locations = self.door_locations[:n]
        self.wall_locations = self.wall_locations[:n]

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]
        self.traj_len = self.actions.shape[1]
        if normalize_action:
            self.action_mean = self.actions.mean(dim=(0, 1))
            self.action_std = self.actions.std(dim=(0, 1))
            self.state_mean = self.states.mean(dim=(0, 1))
            self.state_std = self.states.std(dim=(0, 1))
            self.proprio_mean = self.proprios.mean(dim=(0, 1))
            self.proprio_std = self.proprios.std(dim=(0, 1))
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std

    def get_seq_length(self, idx):
        return self.traj_len

    def get_all_actions(self):
        result = []
        for i in range(len(self.states)):
            T = self.get_seq_length(i)
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        obs_dir = self.data_path / "obses"
        image = torch.load(obs_dir / f"episode_{idx:03d}.pth")
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        door_location = self.door_locations[idx, frames]
        wall_location = self.wall_locations[idx, frames]

        image = image[frames] / 255 
        if self.transform:
            image = self.transform(image)
        obs = {"visual": image,"proprio": proprio}
        return obs, act, state, {'fix_door_location': door_location[0], 'fix_wall_location': wall_location[0]}

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return self.states.shape[0] if not self.n_rollout else self.n_rollout
    
    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return imgs

class WallObjectDataset(WallDataset):
    def __init__(
        self,
        data_path: str = "data/cwall_single",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale=1.0,
        num_clusters=4,
        num_features=3,
        encoder=None,
        use_coord=False,
        use_patch_info=False,
    ):
        super().__init__(
            data_path=data_path,
            n_rollout=n_rollout,
            transform=transform,
            normalize_action=normalize_action,
            action_scale=action_scale,
        )

        self.num_clusters = num_clusters
        self.encoder = encoder
        self.num_features = num_features
        self.use_coord = use_coord
        self.use_patch_info = use_patch_info


    def get_frames(self, idx, frames):
        z_dir = self.data_path / f"objects_{self.encoder}_{self.num_clusters}_feat{self.num_features}"
        z = np.load(z_dir / f"episode_{idx:05d}.npy")
        z = torch.tensor(z)

        if self.use_coord:
            pos_dir = z_dir / f"coord_{idx:05d}.npy"
            pos = np.load(pos_dir)
            pos = torch.tensor(pos)
        
        if self.use_patch_info:
            pos_dir = z_dir / f"patch_{idx:05d}.npy"
            pos = np.load(pos_dir)
            pos = torch.tensor(pos).to(torch.float32)

        obs_dir = self.data_path / "obses"
        image = torch.load(obs_dir / f"episode_{idx:03d}.pth")
        image = image[frames] / 255 
        if self.transform:
            image = self.transform(image)


        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        door_location = self.door_locations[idx, frames]
        wall_location = self.wall_locations[idx, frames]

        obs = {'z': z, 'proprio': proprio, 'visual': image, 'pos': pos}
        return obs, act, state, {'fix_door_location': door_location[0], 'fix_wall_location': wall_location[0]}


def load_wall_slice_train_val(
    transform,
    n_rollout=50,
    data_path='data/wall_single',
    normalize_action=False,
    split_ratio=0.8,
    split_mode="random",
    num_hist=0,
    num_pred=0,
    frameskip=0,
    object=False,
    encoder=None,
    num_clusters=4,
    num_features=3,
    use_coord=False,
    use_patch_info=False,
):  
    if split_mode == "random":
        if object:
            dset = WallObjectDataset(
                n_rollout=n_rollout,
                transform=transform,
                data_path=data_path,
                normalize_action=normalize_action,
                num_clusters=num_clusters,
                num_features=num_features,
                encoder=encoder,
                use_coord=use_coord,
                use_patch_info=use_patch_info,
            )
        else:
            dset = WallDataset(
                n_rollout=n_rollout,
                transform=transform,
                data_path=data_path,
                normalize_action=normalize_action,
            )
        dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
            traj_dataset=dset, 
            train_fraction=split_ratio, 
            num_frames=num_hist + num_pred, 
            frameskip=frameskip
        )
    elif split_mode == "folder":
        if object:
            dset_train = WallObjectDataset(
                n_rollout=n_rollout,
                transform=transform,
                data_path=data_path + "/train",
                normalize_action=normalize_action,
                num_clusters=num_clusters,
                num_features=num_features,
                encoder=encoder,
                use_coord=use_coord,
                use_patch_info=use_patch_info,
            )
            dset_val = WallObjectDataset(
                n_rollout=n_rollout,
                transform=transform,
                data_path=data_path + "/val",
                normalize_action=normalize_action,
                num_clusters=num_clusters,
                num_features=num_features,
                encoder=encoder,
                use_coord=use_coord,
                use_patch_info=use_patch_info,
            )
        else:
            dset_train = WallDataset(
                n_rollout=n_rollout,
                transform=transform,
                data_path=data_path + "/train",
                normalize_action=normalize_action,
            )
            dset_val = WallDataset(
                n_rollout=n_rollout,
                transform=transform,
                data_path=data_path + "/val",
                normalize_action=normalize_action,
            )
        num_frames = num_hist + num_pred
        train_slices = TrajSlicerDataset(dset_train, num_frames, frameskip)
        val_slices = TrajSlicerDataset(dset_val, num_frames, frameskip)

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
