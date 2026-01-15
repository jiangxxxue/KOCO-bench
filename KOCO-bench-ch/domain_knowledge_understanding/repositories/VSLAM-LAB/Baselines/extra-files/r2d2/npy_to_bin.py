import numpy as np

# Load the .npy file
npy_file = '/home/fontan/VSLAM-LAB/VSLAM-Baselines/r2d2/2989.060813 (1).png/descriptors.npy'
data = np.load(npy_file)
print(data)
print(len(data))
print(len(data[0]))

loaded_data = np.fromfile('/home/fontan/VSLAM-LAB/VSLAM-Baselines/r2d2/info.bin', dtype=data.dtype)
print(loaded_data)
print(len(loaded_data))
print(len(loaded_data[0]))
# Open the .bin file in binary write mode
#bin_file = 'output_file.bin'
#with open(bin_file, 'wb') as f:
    # Write the data to the binary file
    #data.tofile(f)

#print(f"Successfully converted {npy_file} to {bin_file}")
