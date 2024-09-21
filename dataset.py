import os
import re

import numpy as np

from torch.utils.data import Dataset

class SimulationDataset(Dataset):

    def __init__(self, data_dir: str):

        self.data_dir = data_dir

        # The number of timesteps and ramp directions of the simulation
        self.timesteps = []
        self.ramp_directions = []

        # Explore the data and extract the files
        # Row index: temperature
        # Column index: ramp direction (0: down, 1: up)
        # Entry: list of .npz files of simulations corresponding to those settings
        self._explore_data_dir()

        # Helper function used in __getitem__
        self.cumulative_lengths = self._calculate_cumulative_lengths()

    def _timesteps_to_data_mapping_index(self, timestep: int):
        assert timestep in self.timesteps, f"timestamp not in data"
        return self.timesteps.index(timestep)

    def _ramp_direction_to_data_mapping_index(self, ramp_direction: str):
        assert ramp_direction in self.ramp_directions, f"ramp direction not recognized"
        if ramp_direction == "ramp_up":
            return 1
        if ramp_direction == "ramp_down":
            return 0

    def _extract_info(self, filepath: str):
        # Regular expression pattern
        pattern = r'(\d+)T_(ramp_up|ramp_down)'
        
        # Try to match the pattern
        match = re.match(pattern, filepath)
        
        if match:
            t_value = match.group(1)
            ramp_direction = match.group(2)
            
            return int(t_value), ramp_direction
        else:
            return None, None

    def _explore_data_dir(self):
        self.data_mapping = []
        # Find the different temp and ramp directions used in the simulations
        for root, dirs, files in os.walk(self.data_dir):
            for dir_ in dirs:
                timestep, ramp_direction = self._extract_info(dir_)
                self.timesteps.append(timestep) if timestep not in self.timesteps else None
                self.ramp_directions.append(ramp_direction) if ramp_direction not in self.ramp_directions else None

        # For each timestep and ramp dir, put all simulations file paths in data mapping
        for timestep in self.timesteps:
            # Add a row for the timestep and all columns for the ramp directions
            self.data_mapping.append([[], []])
            
            # Get the index of the data mapping for the timesteps and ramp direction
            timestep_index = self._timesteps_to_data_mapping_index(timestep)

            for ramp_direction in self.ramp_directions:
                # Get the index of the data mapping for the ramp direction
                ramp_index = self._ramp_direction_to_data_mapping_index(ramp_direction)

                # Walk through the simulations and save filepath of npz files
                simulation_dir = os.path.join(self.data_dir, f"{timestep}T_{ramp_direction}")
                for root, dirs, files in os.walk(simulation_dir):
                    # Save the file paths
                    file_paths = [os.path.join(simulation_dir, file) for file in files]
                    self.data_mapping[timestep_index][ramp_index] = file_paths

    def _calculate_cumulative_lengths(self):
        """
        Calculate the cumulative lengths of files for each temperature and ramp direction.

        This method iterates through the data_mapping structure, which is organized as follows:
        - Each row represents a temperature
        - Each row has two columns: [ramp_down_files, ramp_up_files]
        - Each cell contains a list of filenames

        Returns:
        list: A list where each element represents the cumulative count of files
            up to and including the corresponding temperature. The first element
            is always 0, representing the starting point.

        Example:
        If data_mapping = [
            [[file1, file2], [file3, file4, file5]],  # Temperature 1
            [[file6, file7], [file8]]                 # Temperature 2
        ]
        Then this method will return [0, 5, 8]
        """
        cumulative = [0]  # Start with 0 as the initial cumulative count
        
        for temp_row in self.data_mapping:
            # Calculate the total number of files for this temperature
            # by summing the lengths of ramp_up and ramp_down lists
            files_in_temp = len(temp_row[0]) + len(temp_row[1])
            
            # Add this to the previous cumulative count
            cumulative.append(cumulative[-1] + files_in_temp)
        
        return cumulative

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find the temperature (row)
        temp_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        
        # Adjust idx for the found temperature
        idx -= self.cumulative_lengths[temp_idx]
        
        # Determine ramp direction and file index
        ramp_down_length = len(self.data_mapping[temp_idx][0])
        if idx < ramp_down_length:
            ramp_direction = 0  # down
            file_idx = idx
        else:
            ramp_direction = 1  # up
            file_idx = idx - ramp_up_length

        # Get the filepath
        filepath = self.data_mapping[temp_idx][ramp_direction][file_idx]

        # fetch data
        data = np.load(filepath)

        # return X, Y data
        return data["output"], data["forcing"]

    def channel_name_to_index(self, type_: str, name: str) -> int:
        """ Given the type of the data and the name, 
            returns the index of the channel in the numpy array.
        """
        assert type_ in ["output", "forcing"], f"Provided type {type_} is not in available {['output', 'forcing']}"

        mapping = {
            "forcing":{
                "density": 0,
                "ion_velocity": 1,
                "atom_velocity": 2,
                "temperature": 3,
                "atom_density": 4,
                "molecule_density": 5,
            },
            "output": {
                "particle_forcing": 0,
                "heat_forcing": 1
            }
        }

        try:
            return mapping[type_][name]
        except KeyError:
            raise KeyError(f"Channel {name} does not exist for type {type_}")

    def index_to_channel_name(self, type_: str, idx: int) -> int:
        """ Given the type of the data and the index of its column, 
            provide the name of the channel.
        """
        assert type_ in ["output", "forcing"], f"Provided type {type_} is not in available {['output', 'forcing']}"

        mapping = {
            "forcing":{
                0: "density",
                1: "ion_velocity",
                2: "atom_velocity",
                3: "temperature",
                4: "atom_density",
                5: "molecule_density",
            },
            "output": {
                0: "particle_forcing",
                1: "heat_forcing"
            }
        }

        try:
            return mapping[type_][idx]
        except KeyError:
            raise KeyError(f"Idx {idx} does not exist for type {type_}")