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
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.data_list_file = os.path.join(self.data_root, f"{self.subset}.txt")

        print(f"[DATASET] Open file {self.data_list_file}")
        with open(self.data_list_file, "r") as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split("-")[0]
            model_id = line.split("-")[1].split(".")[0]
            self.file_list.append(
                {"taxonomy_id": taxonomy_id, "model_id": model_id, "file_path": line}
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
        data, _, _ = Dynamics.pc_norm(data)
        info = {
            "vis_mask": vis_mask,
        }

        return sample["taxonomy_id"], sample["model_id"], data, info

    def __len__(self):
        return len(self.file_list)
