from RobustMethodPlots import RobustCylinderFitting
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def load_point_cloud(file_path, intensity_threshold = 0, full_data = False):
    """
    Load a point cloud from LAS or LAZ files.

    Parameters:
    file_path : str
        Path to the LAS or LAZ file.

    Returns:
    point_cloud : ndarray
        Nx3 matrix of point coordinates (x, y, z).
    """
    if ".xyz" in file_path:
        # Load point cloud from an XYZ file
        point_data = np.loadtxt(file_path, dtype=np.float64)
        if point_data.shape[1] == 3:
            point_cloud = point_data
        elif point_data.shape[1] == 4:
            I = point_data[:, 3] > intensity_threshold
            point_cloud = point_data[I, :3]
        else:
            raise ValueError("Unsupported format in XYZ file.")
        return point_cloud if not full_data else (point_cloud, point_data)

    return point_cloud if not full_data else (point_cloud,point_data)



if __name__ == "__main__":
    fitter = RobustCylinderFitting()
    with PdfPages("multiple_plots.pdf") as pdf:
        fig = plt.figure(figsize=(8, 10))
        fig_in = fig
        for num, branch_file in enumerate(os.listdir("segments")):
            
            P = load_point_cloud(f"segments/{branch_file}")
            x, y, r, fig_out = fitter.fit(P, num, fig_in)
            fig_in = fig_out
            if (num + 1)%3 == 0:
                pdf.savefig()
                fig_in = plt.figure(figsize=(8, 10))


