# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import glob
import numpy as np
import torch
from torch_geometric.data import Dataset, Data

__all__ = ["Custom"]


class Custom(Dataset):
    r"""Custom Dataset to manage loading coordinates, embedding indices,
    energies and forces from NumPy files. :obj:`coordglob` and :obj:`embedglob`
    are required parameters. Either :obj:`energyglob`, :obj:`forceglob` or both
    must be given as targets.

    Args:
        coordglob (string): Glob path for coordinate files. Stored as "pos".
        embedglob (string): Glob path for embedding index files. Stored as "z" (atomic number).
        energyglob (string, optional): Glob path for energy files. Stored as "y".
            (default: :obj:`None`)
        forceglob (string, optional): Glob path for force files. Stored as "neg_dy".
            (default: :obj:`None`)
        preload_memory_limit (int, optional): If the dataset is smaller than this limit (in MB), preload it into CPU memory.
        transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in an :obj:`torch_geometric.data.Data` object and returns a transformed version. The data object will be transformed before being saved to disk.
        pre_filter (callable, optional): A function that takes in an :obj:`torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset.


    Example:
        >>> data = Custom(coordglob="coords_files*npy", embedglob="embed_files*npy")
        >>> sample = data[0]
        >>> assert hasattr(sample, "pos"), "Sample doesn't contain coords"
        >>> assert hasattr(sample, "z"), "Sample doesn't contain atom numbers"

    Notes:
        For each sample in the data:
              - "pos" is an array of shape (n_atoms, 3)
              - "z" is an array of shape (n_atoms,).
              - If present, "y" is an array of shape (1,) and "neg_dy" has shape (n_atoms, 3)
    """

    def __init__(
        self,
        coordglob,
        embedglob,
        energyglob=None,
        forceglob=None,
        extcoordglob=None,
        extchargeglob=None,
        espglob=None,
        espgradglob=None,
        preload_memory_limit=1024,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(None, transform, pre_transform, pre_filter)
        assert energyglob is not None or forceglob is not None, (
            "Either energies, forces or both must " "be specified as the target"
        )
        self.has_energies = energyglob is not None
        self.has_forces = forceglob is not None
        self.has_extcoord = extcoordglob is not None
        self.has_extcharge = extchargeglob is not None
        self.has_esp = espglob is not None
        self.has_espgrad = espgradglob is not None
        self.fields = [
            ("pos", "pos", torch.float32),
            ("z", "types", torch.long),
        ]
        self.files = {}
        self.files["pos"] = sorted(glob.glob(coordglob))
        self.files["z"] = sorted(glob.glob(embedglob))
        assert len(self.files["pos"]) == len(self.files["z"]), (
            f"Number of coordinate files {len(self.files['pos'])} "
            f"does not match number of embed files {len(self.files['z'])}."
        )
        if self.has_energies:
            self.files["y"] = sorted(glob.glob(energyglob))
            self.fields.append(("y", "energy", torch.float32))
            assert len(self.files["pos"]) == len(self.files["y"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of energy files {len(self.files['y'])}."
            )
        if self.has_forces:
            self.files["neg_dy"] = sorted(glob.glob(forceglob))
            self.fields.append(("neg_dy", "forces", torch.float32))
            assert len(self.files["pos"]) == len(self.files["neg_dy"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of force files {len(self.files['neg_dy'])}."
            )
        if self.has_extcoord:
            self.files["ext_pos"] = sorted(glob.glob(extcoordglob))
            self.fields.append(("ext_pos", "ext_pos", torch.float32))
            assert len(self.files["pos"]) == len(self.files["ext_pos"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of ext coordinate files {len(self.files['ext_pos'])}."
            )
        if self.has_extcharge:
            self.files["ext_charge"] = sorted(glob.glob(extchargeglob))
            self.fields.append(("ext_charge", "ext_charge", torch.float32))
            assert len(self.files["pos"]) == len(self.files["ext_charge"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of ext charge files {len(self.files['ext_charge'])}."
            )
        if self.has_esp:
            self.files["esp"] = sorted(glob.glob(espglob))
            self.fields.append(("esp", "esp", torch.float32))
            assert len(self.files["pos"]) == len(self.files["esp"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of esp files {len(self.files['esp'])}."
            )
        if self.has_espgrad:
            self.files["esp_grad"] = sorted(glob.glob(espgradglob))
            self.fields.append(("esp_grad", "esp_grad", torch.float32))
            assert len(self.files["pos"]) == len(self.files["esp_grad"]), (
                f"Number of coordinate files {len(self.files['pos'])} "
                f"does not match number of esp gradient files {len(self.files['esp_grad'])}."
            )
        print("Number of files: ", len(self.files["pos"]))
        self.cached = False
        total_data_size = self._initialize_index()
        print(f"Combined dataset size {len(self.index)}")
        # If the dataset is small enough, load it whole into CPU memory
        data_size_limit = preload_memory_limit * 1024 * 1024
        if total_data_size < data_size_limit:
            self.cached = True
            print(
                f"Preloading Custom dataset (of size {total_data_size / 1024**2:.2f} MB)"
            )
            self._preload_data()
        else:
            self._store_numpy_memmaps()

    def _preload_data(self):
        """Load the input files as Torch tensors.
        Each file can have different number of atoms, so each one is stored in a different tensor.
        """
        self.stored_data = {}
        self.stored_data["pos"] = [
            torch.from_numpy(np.load(f)) for f in self.files["pos"]
        ]
        self.stored_data["z"] = [
            torch.from_numpy(np.load(f).astype(int))
            .unsqueeze(0)
            .expand(self.stored_data["pos"][i].shape[0], -1)
            for i, f in enumerate(self.files["z"])
        ]
        if self.has_energies:
            self.stored_data["y"] = [
                torch.from_numpy(np.load(f)) for f in self.files["y"]
            ]
        if self.has_forces:
            self.stored_data["neg_dy"] = [
                torch.from_numpy(np.load(f)) for f in self.files["neg_dy"]
            ]
        if self.has_extcoord:
            self.stored_data["ext_pos"] = [
                torch.from_numpy(np.load(f)) for f in self.files["ext_pos"]
            ]
        if self.has_extcharge:
            self.stored_data["ext_charge"] = [
                torch.from_numpy(np.load(f)) for f in self.files["ext_charge"]
            ]
        if self.has_esp:
            self.stored_data["esp"] = [
                torch.from_numpy(np.load(f)) for f in self.files["esp"]
            ]
        if self.has_espgrad:
            self.stored_data["esp_grad"] = [
                torch.from_numpy(np.load(f)) for f in self.files["esp_grad"]
            ]

    def _store_numpy_memmaps(self):
        """Create a dictionary with numpy arrays with mmap_mode="r" for each file."""
        self.stored_data = {}
        self.stored_data["pos"] = [np.load(f, mmap_mode="r") for f in self.files["pos"]]
        self.stored_data["z"] = []
        for i, f in enumerate(self.files["z"]):
            loaded_data = np.load(f).astype(int)
            desired_shape = (len(self.stored_data["pos"][i]), loaded_data.shape[0])
            broadcasted_data = np.broadcast_to(
                loaded_data[np.newaxis, :], desired_shape
            )
            self.stored_data["z"].append(broadcasted_data)
        if self.has_energies:
            self.stored_data["y"] = [np.load(f, mmap_mode="r") for f in self.files["y"]]
        if self.has_forces:
            self.stored_data["neg_dy"] = [
                np.load(f, mmap_mode="r") for f in self.files["neg_dy"]
            ]
        if self.has_extcoord:
            self.stored_data["ext_pos"] = [
                np.load(f, mmap_mode="r") for f in self.files["ext_pos"]
            ]
        if self.has_extcharge:
            self.stored_data["ext_charge"] = [
                np.load(f, mmap_mode="r") for f in self.files["ext_charge"]
            ]
        if self.has_esp:
            self.stored_data["esp"] = [
                np.load(f, mmap_mode="r") for f in self.files["esp"]
            ]
        if self.has_espgrad:
            self.stored_data["esp_grad"] = [
                np.load(f, mmap_mode="r") for f in self.files["esp_grad"]
            ]

    def _initialize_index(self):
        """Initialize the index for the dataset.
        The index relates a sample to the file it belongs to and the index within that file.
        Returns:
            int: Total size of the dataset in bytes.
        """
        # create index
        self.index = []
        nfiles = len(self.files["pos"])
        total_data_size = 0  # Number of bytes in the dataset
        for i in range(nfiles):
            coord_data = np.load(self.files["pos"][i], mmap_mode="r")
            embed_data = np.load(self.files["z"][i]).astype(int)
            size = coord_data.shape[0]
            total_data_size += coord_data.nbytes + embed_data.nbytes
            self.index.extend(list(zip([i] * size, range(size))))
            # consistency check
            assert coord_data.shape[1] == embed_data.shape[0], (
                f"Number of atoms in coordinate file {i} ({coord_data.shape[1]}) "
                f"does not match number of atoms in embed file {i} ({embed_data.shape[0]})."
            )
            if self.has_energies:
                energy_data = np.load(self.files["y"][i], mmap_mode="r")
                total_data_size += energy_data.nbytes
                assert coord_data.shape[0] == energy_data.shape[0], (
                    f"Number of frames in coordinate file {i} ({coord_data.shape[0]}) "
                    f"does not match number of frames in energy file {i} ({energy_data.shape[0]})."
                )
            if self.has_forces:
                force_data = np.load(self.files["neg_dy"][i], mmap_mode="r")
                total_data_size += force_data.nbytes
                assert coord_data.shape == force_data.shape, (
                    f"Data shape of coordinate file {i} {coord_data.shape} "
                    f"does not match the shape of force file {i} {force_data.shape}."
                )
            if self.has_extcoord:
                extcoord_data = np.load(self.files["ext_pos"][i], mmap_mode="r")
                total_data_size += extcoord_data.nbytes
                assert coord_data.shape[0] == extcoord_data.shape[0], (
                    f"Number of frames in coordinate file {i} ({coord_data.shape[0]}) "
                    f"does not match number of frames in ext coordinate file {i} ({extcoord_data.shape[0]})."
                )
            if self.has_extcharge:
                extcharge_data = np.load(self.files["ext_charge"][i], mmap_mode="r")
                total_data_size += extcharge_data.nbytes
                assert coord_data.shape[0] == extcharge_data.shape[0], (
                    f"Number of frames in coordinate file {i} ({coord_data.shape[0]}) "
                    f"does not match number of frames in ext charge file {i} ({extcharge_data.shape[0]})."
                )
            if self.has_esp:
                esp_data = np.load(self.files["esp"][i], mmap_mode="r")
                total_data_size += esp_data.nbytes
                assert coord_data.shape[0] == esp_data.shape[0], (
                    f"Number of frames in coordinate file {i} ({coord_data.shape[0]}) "
                    f"does not match number of frames in esp file {i} ({esp_data.shape[0]})."
                )
            if self.has_espgrad:
                espgrad_data = np.load(self.files["esp_grad"][i], mmap_mode="r")
                total_data_size += espgrad_data.nbytes
                assert coord_data.shape[0] == espgrad_data.shape[0], (
                    f"Number of frames in coordinate file {i} {coord_data.shape} "
                    f"does not match number of frames in esp gradient file {i} {espgrad_data.shape}."
                )
        return total_data_size

    def get(self, idx):
        fileid, index = self.index[idx]
        data = Data()
        for field in self.fields:
            # The dataset is stored as mem mapped numpy arrays unless it is cached,
            # in which case it is already stored as torch tensors
            f = self.stored_data[field[0]][fileid][index]
            data[field[0]] = f if self.cached else torch.from_numpy(np.array(f))
        return data

    def len(self):
        return len(self.index)
