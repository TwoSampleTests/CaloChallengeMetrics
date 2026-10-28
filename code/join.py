import h5py
import numpy as np
import os




def extract_shower_and_energy(file_handle):
    """Reads .hdf5 file and returns showers and incident energies."""
    showers = file_handle['showers'][:]
    energies = file_handle['incident_energies'][:]
    return showers, energies

reference_file_1 = "/teo_fs_fast/projects/GENOVA/CaloChallenge/Dataset_3/data/dataset_3_3.hdf5"
reference_file_2 = "/teo_fs_fast/projects/GENOVA/CaloChallenge/Dataset_3/data/dataset_3_4.hdf5"

output_file = "/teo_fs_fast/projects/GENOVA/CaloChallenge/Dataset_3/data/dataset_3_full.hdf5"

# ---- load first file ----
with h5py.File(reference_file_1, "r") as source_file:
    showers_1, energies_1 = extract_shower_and_energy(source_file)

# ---- load second file ----
with h5py.File(reference_file_2, "r") as source_file:
    showers_2, energies_2 = extract_shower_and_energy(source_file)

# ---- concatenate datasets ----
showers = np.concatenate([showers_1, showers_2], axis=0)
incident_energies = np.concatenate([energies_1, energies_2], axis=0)

print("Merged dataset shape:")
print("showers:", showers.shape)
print("incident_energies:", incident_energies.shape)

# ---- write new hdf5 file ----
with h5py.File(output_file, "w") as f:
    
    f.create_dataset(
        "showers",
        data=showers,
        compression="gzip"
    )
    
    f.create_dataset(
        "incident_energies",
        data=incident_energies,
        compression="gzip"
    )

print(f"Created merged file: {output_file}")