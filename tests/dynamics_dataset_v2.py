import os
import torch
import numpy as np
import torch.utils.data as data
import logging
import pickle
from pathlib import Path
# NOTE: Most of this code is copied from FFN_hist with minor changes only in get_tuple

class FFN_dynamics_v2(data.Dataset):
    def __init__(self, config):
        self.data_root = Path(config.DATA_PATH)
        self.pc_root = Path(config.PC_PATH)
        self.subset = config.subset
        self.data_list_file = self.data_root / f"{self.subset}.txt"
        self.ver  = config.ver

        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = 0  # always cloth
            model_id = line
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": self.pc_root / Path(line)}
            )
        print(f"[DATASET] {len(self.file_list)} instances were loaded")

    @staticmethod
    def pc_norm(pc_before, pc_after):
        """pc: NxC, return NxC"""
        centroid = pc_after.mean(axis=0)
        pc_before = (pc_before - centroid) / 1
        pc_after = (pc_after - centroid) / 1
        #Added a fixed z scale
        pc_before[:,-1] = pc_before[:,-1]*10
        pc_after[:,-1] = pc_after[:,-1]*10
        return pc_before, pc_after, 1, centroid

    def load_pc(self, pth, prefix="curr"):
        # Load pc
        pc = np.load(pth / f"{prefix}_object_pcd.npy").astype(np.float32)
        return pc

    def get_tuple(self, pth):
        """Load tuple of data (obs, action, next_obs) from the folder specified by model_id."""
        if self.ver == "normal":
            pc_obs = self.load_pc(pth, prefix="prev")
            pc_nobs = self.load_pc(pth, prefix="curr")
            pc_obs = pc_obs[:, [0,2,1]]
            pc_nobs = pc_nobs[:, [0,2,1]]
            # load action
            pc_obs, pc_nobs, scale, center = FFN_dynamics_v2.pc_norm(pc_obs, pc_nobs)  # both n x 3
            act_loc = np.load(pth / 'action_location.npy')
            action_flow = np.load(pth / 'action_param.npy')
            max_scale = [0.125, 0.125, 0.125]
            action_scaled = action_flow.copy()
            action_scaled[0] *= max_scale[0]
            action_scaled[1] = ((action_scaled[1] + 1) / 2) * max_scale[1] # rescaled [-1, 1] to [0, max]
            action_scaled[2] *= max_scale[2]
            min_idx = ((act_loc - pc_obs)**2).sum(axis=1).argmin()
            dyn_input = np.zeros(pc_obs.shape)
            dyn_input[min_idx, :] = action_scaled
            dyn_input = np.concatenate([pc_obs, dyn_input], axis=-1)
            dyn_output = pc_nobs - pc_obs

            return (
                pc_obs,
                pc_nobs,
                dyn_input,
                dyn_output
            )
        
        if self.ver == "9D":
            pc_obs = self.load_pc(pth, prefix="prev")
            pc_nobs = self.load_pc(pth, prefix="curr")
            pc_obs = pc_obs[:, [0,2,1]]
            pc_nobs = pc_nobs[:, [0,2,1]]
            # load action
            pc_obs, pc_nobs, scale, center = FFN_dynamics_v2.pc_norm(pc_obs, pc_nobs)  # both n x 3
            act_loc = np.load(pth / 'action_location.npy')
            action_flow = np.load(pth / 'action_param.npy')
            max_scale = [0.125, 0.125, 0.125]
            action_scaled = action_flow.copy()
            action_scaled[0] *= max_scale[0]
            action_scaled[1] = ((action_scaled[1] + 1) / 2) * max_scale[1] # rescaled [-1, 1] to [0, max]
            action_scaled[2] *= max_scale[2]
            min_idx = ((act_loc - pc_obs)**2).sum(axis=1).argmin()
            act_arr = np.ones(pc_obs.shape) * action_scaled
            act_loc_arr = pc_obs - pc_obs[min_idx, :]
            dyn_input = np.concatenate([pc_obs, act_loc_arr, act_arr], axis=-1)
            dyn_output = pc_nobs - pc_obs

            return (
                pc_obs,
                pc_nobs,
                dyn_input,
                dyn_output
            )
        

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        (
            pc_obs,
            pc_nobs,
            dyn_input,
            dyn_output
        ) = self.get_tuple(sample["file_path"])

        data = {
            "pc_obs": pc_obs,
            "pc_nobs": pc_nobs,
            "dyn_input":dyn_input,
            "dyn_output":dyn_output
        }

        return sample["taxonomy_id"], sample["model_id"], data, {}

    def __len__(self):
        return len(self.file_list)

if __name__ == "__main__":
    ds = FFN_dynamics_v2()