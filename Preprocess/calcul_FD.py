import sys
import numpy as np

MC_file=sys.argv[1]
FD_file=sys.argv[2]

mc_data = np.loadtxt(MC_file)
fd_data = mc_data[1:, :] - mc_data[0:-1, :]
fd_data[:, :3] *= 35
fd_data = np.abs(fd_data)
fd_data = fd_data.sum(axis=1)
fd_data = np.insert(fd_data, 0, 0)

np.savetxt(FD_file, fd_data, fmt="%.7f")