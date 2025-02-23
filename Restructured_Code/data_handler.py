import pandas as pd
import scipy.io as sio
import os
import numpy as np
import csv
import json

class DataHandler:
    def __init__(self,base_dir="./data"):
        self.base_dir = base_dir
        self.directories={
            "parameters": os.path.join(base_dir,"parameters"),
            "trajectories": os.path.join(base_dir,"trajectories"),
            "densities": os.path.join(base_dir,"densities"),
            "average_trajectories": os.path.join(base_dir,"average_trajectories"),
            "counts": os.path.join(base_dir,"counts"),
            "plots": os.path.join(base_dir,"plots"),
            "dominance_statistics": os.path.join(base_dir,"dominance_statistics"),
            "joch_data": os.path.join(base_dir,"joch_data")
        }

        self.create_directories()

    def create_directories(self):
        for directory in self.directories.values():
            os.makedirs(directory,exist_ok=True)

    def _get_path(self, category, filename, ext):
        """Generates the correct file path for a given category and extension."""
        if category not in self.directories:
            raise ValueError(f"Invalid category: {category}. Choose from {list(self.directories.keys())}")
        return os.path.join(self.directories[category], f"{filename}.{ext}")

    def save_npy(self, category, filename, data):
        """Saves a numpy array to a file."""
        path = self._get_path(category, filename, "npy")
        np.save(path, data)
        print(f"Saved {category}/{filename}.npy")

    def load_npy(self, category, filename):
        """Loads a numpy array from a file."""
        path = self._get_path(category, filename, "npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = np.load(path, allow_pickle=True)
        print(f"Loaded {category}/{filename}.npy")
        return data

    def save_npz(self, category, filename, **data):
        """Saves multiple numpy arrays to a single file."""
        path = self._get_path(category, filename, "npz")
        np.savez(path, **data)
        print(f"Saved {category}/{filename}.npz")

    def load_npz(self, category, filename):
        """Loads multiple numpy arrays from a single file."""
        path = self._get_path(category, filename, "npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = np.load(path)
        print(f"Loaded {category}/{filename}.npz")
        return data

    def save_mat(self, category, filename, data):
        """Saves data to a MATLAB file."""
        path = self._get_path(category, filename, "mat")
        sio.savemat(path, data)
        print(f"Saved {category}/{filename}.mat")

    def load_mat(self, category, filename):
        """Loads data from a MATLAB file."""
        path = self._get_path(category, filename, "mat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        data = sio.loadmat(path)
        print(f"Loaded {category}/{filename}.mat")
        return data

    def save_csv(self, category, filename, data):
        """Saves data to a CSV file."""
        path = self._get_path(category, filename, "csv")
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False,header=True)
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
        print(f"Saved {category}/{filename}.csv")

    def load_csv(self, category, filename):
        """Loads data from a CSV file."""
        path = self._get_path(category, filename, "csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        print(f"Loaded {category}/{filename}.csv")
        return df

    def save_json(self, category, filename, data):
        """Saves data to a JSON file."""
        path = self._get_path(category, filename, "json")
        with open(path, "w") as file:
            json.dump(data, file, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o))
        print(f"Saved {category}/{filename}.json")

    def load_json(self, category, filename):
        """Loads data from a JSON file."""
        path = self._get_path(category, filename, "json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r") as file:
            data = json.load(file)
        print(f"Loaded {category}/{filename}.json")
        return data

    def save_fig(self, category, filename, fig):
        """Saves a matplotlib figure to a file."""
        path = self._get_path(category, filename, "png")
        fig.savefig(path)
        print(f"Saved {category}/{filename}.png")





