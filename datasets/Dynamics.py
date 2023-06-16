import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import logging
import pickle
from pathlib import Path


@DATASETS.register_module()
class Dynamics(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = Path(config.PC_PATH)
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")

        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = 0
            model_id = 0
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": self.pc_path / line}
            )
        print(f"[DATASET] {len(self.file_list)} instances were loaded")

    @staticmethod
    def pc_norm(pc):
        """pc: NxC, return NxC"""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        # pc = pc / m
        pc[:,-1] = pc[:,-1] * 1.0
        return pc, centroid, 1

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        file_path = Path(sample['file_path'])
        data = np.load(file_path/ "curr_object_pcd.npy").astype(np.float32)
        vis_mask = np.load(file_path / "curr_visibility_mask.npy").astype(np.float32)
        data = data[:, :3]
        data_norm, _ , _ = Dynamics.pc_norm(data)
        
        pc_obs = np.load(file_path / f"prev_object_pcd.npy").astype(np.float32)
        pc_obs = pc_obs[:, [0,2,1]]
        pc_nobs = data[:, [0,2,1]] 
        pc_nobs, centroid, _ = Dynamics.pc_norm(pc_nobs)
        pc_obs = pc_obs - centroid
        pc_obs[:,-1] = pc_obs[:,-1] * 1.0
        act_loc = np.load(file_path / 'action_location.npy')
        action_flow = np.load(file_path / 'action_param.npy')
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
        
        info = {
            "vis_mask": vis_mask,
            "dyn_input": dyn_input,
            "dyn_output": dyn_output,
        }
        return sample["taxonomy_id"], sample["model_id"], data_norm, info

    def __len__(self):
        return len(self.file_list)
