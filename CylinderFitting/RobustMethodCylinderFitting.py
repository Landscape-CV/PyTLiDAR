"""
Implementation in Python of this paper: https://www.sciencedirect.com/science/article/pii/S0263224119301046
Robust cylinder fitting in laser scanning point cloud data
"""
from robpy.covariance import DetMCD
import rpca.ealm
import laspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import rpca.ialm  # Needed for 3D plotting
from sklearn.covariance import MinCovDet 
import ltsfit # https://pypi.org/project/ltsfit/ -- need to cite in paper. 
import math
from circle_fit import hyperSVD


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
    with laspy.open(file_path) as las:
        point_data = las.read()
        point_data = np.vstack((point_data.x, point_data.y, point_data.z,point_data.intensity)).T.astype('float64')
        I = point_data[:,3]>intensity_threshold
        point_data = point_data[I]
        point_cloud = point_data[:,0:3]
    return point_cloud if not full_data else (point_cloud,point_data)

def plot_cylinder(cylinder, line1,line2=None,line3=None, linescale = 1):
    """
    Plots the cylinder for debugging.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cylinder[:, 0], cylinder[:, 1], cylinder[:, 2], color='green')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Scatter Plot')
    fig.axes


    ax.plot([0, line1[0]*linescale], [0, line1[1]*linescale], [0, line1[2]*linescale], color='red', linewidth=2)
    

    ax.plot([0, line2[0]*linescale], [0, line2[1]*linescale], [0, line2[2]*linescale], color='purple', linewidth=2)

    ax.plot([0, line3[0]*linescale], [0, line3[1]*linescale], [0, line3[2]*linescale], color='blue', linewidth=2)

    # Display
    plt.axis('equal')
    plt.show()

def plot_axes(fig, axis_array):
    """
    Plots the axis for debugging. 
    """
    
    
def create_fake_cylinder(length = 10, noise = False):

        # 1. Create a synthetic cylinder
    theta = np.linspace(0, 2 * np.pi, 20)
    z = np.linspace(0, length, 100)
    theta, z = np.meshgrid(theta, z)
    x = np.cos(theta).ravel()
    y = np.sin(theta).ravel()
    z = z.ravel()

    cylinder = np.vstack((x, y, z)).T  # shape (n, 3)

    # 2. Add sparse noise
    np.random.seed(42)
    sparse_noise = np.zeros_like(cylinder)
    num_outliers = 100
    indices = np.random.choice(len(cylinder), size=num_outliers, replace=False)
    sparse_noise[indices] = 0.5 * np.random.randn(num_outliers, 3)
    if noise:
        D = cylinder + sparse_noise
    else:
        D = cylinder#  + sparse_noise


    return D

def generate_noisy_line(
    num_points=500,
    length=10.0,
    direction=np.array([0, 1, 0]),
    origin=np.array([0, 0, 0]),
    noise_std=0.0005,
    random_seed=None
):
    """
    Generate a noisy line-shaped point cloud in 3D space.
    
    Parameters:
    - num_points: Number of points along the line
    - length: Total length of the line
    - direction: Unit vector for the line's direction
    - origin: Starting point of the line
    - noise_std: Standard deviation of Gaussian noise
    - random_seed: For reproducibility
    
    Returns:
    - point_cloud: (N, 3) ndarray
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    direction = direction / np.linalg.norm(direction)  # Normalize direction

    # Generate linearly spaced points along the direction
    t = np.linspace(0, length, num_points)
    clean_line = origin + np.outer(t, direction)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, clean_line.shape)
    point_cloud = clean_line#  + noise

    return point_cloud



class RobustCylinderFitting: 
    def __init__(self):
        pass

    def fit(self, point_cloud):
        """
        Main process step
        """
        # Task 1: get cylinder orientation. 
        point_cloud = self._normalize_pointcloud(point_cloud)
        pc1, pc2, pc3 = self._get_pcs(point_cloud)
        length = self._get_length(point_cloud, pc1)
        
        # Task 2: Robust circle fitting
        x, y, r = self._fit_circle(point_cloud, pc2, pc3)

        return x, y, r


    def _get_cylinder_orientation(self, point_cloud):
        """
        Task 1: Performing RPCA for estimating cylinder orientation


        """
        
        dense_matrix, sparse_matrix = rpca.ialm.fit(point_cloud, verbose=False)
        
        # U, S, Vt = np.linalg.svd(dense_matrix)
        

        U, S, Vt = np.linalg.svd(dense_matrix, full_matrices=False)
        # explained_ratios = (S**2) / np.sum(S**2)
        # for i, ratio in enumerate(explained_ratios):
        #     print(f"PC{i+1}: {ratio:.4f}")

        # principal_axis_index = np.argmax(explained_ratios)
        # principal_axis = Vt[principal_axis_index, :]


        # - U: left singular vectors — shape (m, m)
        # - S: singular values — shape (min(m, n),)
        # - Vt: right singular vectors transposed — shape (n, n)

        first_pc = Vt[0, :] 
        second_pc = Vt[1, :] 
        third_pc = Vt[2, :] 
        # print(first_pc)
        # print(second_pc)
        # print(third_pc)
        # print(pc1.shape)
        # print(pc2.shape)
        # print(pc3.shape)




        return first_pc, second_pc, third_pc

    def _get_pcs(self, point_cloud):
        """
        Task 1: Using MCD instead for cylinder orientation
        """
        
        mcd = DetMCD()
        covariance = mcd.calculate_covariance(point_cloud)

        U, S, Vt = np.linalg.svd(covariance, full_matrices=False)
        print(Vt.shape)
        first_pc = Vt[0, :] 
        second_pc = Vt[1, :] 
        third_pc = Vt[2, :] 
        print(first_pc)
        print(second_pc)
        print(third_pc)
        return first_pc, second_pc, third_pc


    def _normalize_pointcloud(self, pointcloud):
        """
        centers the pointcloud around 0,0,0 
        """
        mean = np.mean(pointcloud, axis=0)
        normalized = pointcloud - mean

        return normalized
    
    def _get_length(self, cylinder, pc1):

        # proj = dense_matrix @ first_pc   # A is the denoised point cloud
        # print(proj.shape)


        projection = np.dot(cylinder, pc1)
        print(projection.shape)
        length = np.max(projection) - np.min(projection)

        # plt.plot(projection)
        # plt.show()
        return length
    
    def _fit_circle(self, cylinder, pc2, pc3):
        # Project points onto the orthogonal plane created by pc2, pc3
        projection1 = np.dot(cylinder, pc2)
        projection2 = np.dot(cylinder, pc3)
        circle_projection = np.array([projection1, projection2])
        circle_projection = np.transpose(circle_projection)
        # print(circle_projection.shape)
        fig, ax = plt.subplots()

        ax.scatter(circle_projection[:, 0], circle_projection[:, 1], s=1)
        ax.axis('equal')
        # Use Hyper circle fitting method. 
        # x, y, r, s = hyperSVD(circle_projection)
        x, y, r = self._RLTS(circle_projection)
        # Plot circle
        print(x, y, r*2)
        circle1 = Circle((x, y), r, edgecolor='blue', facecolor='none', linewidth=2)
        ax.add_patch(circle1)


        x, y, r = self._WRLTS(circle_projection)
        circle2 = Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=1)
        ax.add_patch(circle2)
        plt.show()
        return x, y, r
    
    def _RLTS(self, pointcloud):
        """
        Repeated Least Trimmed Square (RLTS)
        """
        h_0 = 4
        p_r = 0.999
        eps = 0.5
        h = math.ceil(pointcloud.shape[0] * 0.5)

        # Fit initial circle from randomly selected points
        indices = np.random.choice(pointcloud.shape[0], size=h_0, replace=False)
        random_points = pointcloud[indices]

        # Get circle parameters for initial guess
        print(random_points.shape)
        x, y, r, s = hyperSVD(random_points)

        e = self._compute_residuals(point_cloud, x, y, r) # Nx3

        # Then we sort the points according to their residuals. 
        e_sorted_indices = np.argsort(e)
        sorted_pointcloud = pointcloud[e_sorted_indices]
        top_h_points = sorted_pointcloud[:h, :]

        # Then repeat the algorithm. 
        for i in range(100):

            # fig, ax = plt.subplots()
            # ax.scatter(pointcloud[:, 0], pointcloud[:, 1], s=1)
            # ax.axis('equal')
            # circle = Circle((x, y), r, edgecolor='blue', facecolor='none', linewidth=2)
            # ax.add_patch(circle)
            # plt.show()


            x, y, r, s = hyperSVD(top_h_points)
            e = self._compute_residuals(point_cloud, x, y, r) # Nx3
            e_sorted_indices = np.argsort(e)
            sorted_pointcloud = pointcloud[e_sorted_indices]
            top_h_points = sorted_pointcloud[:h, :]

        return x, y, r

    def _WRLTS(self, pointcloud):
        """
        Weighted Repeated Least Trimmed Square (WRLTS)
        """
        h_0 = 4
        p_r = 0.999
        eps = 0.5
        h = math.ceil(pointcloud.shape[0] * 0.5)

        # Fit initial circle from randomly selected points
        indices = np.random.choice(pointcloud.shape[0], size=h_0, replace=False)
        random_points = pointcloud[indices]

        # Get circle parameters for initial guess
        print(random_points.shape)
        x, y, r, s = hyperSVD(random_points)

        e = self._compute_residuals(point_cloud, x, y, r) # Nx3

        # Then we sort the points according to their residuals. 
        e_sorted_indices = np.argsort(e)
        sorted_pointcloud = pointcloud[e_sorted_indices]
        top_h_points = sorted_pointcloud[:h, :]

        # Then repeat the algorithm. 
        for i in range(100):
            x, y, r, s = hyperSVD(top_h_points)
            e = self._compute_residuals(point_cloud, x, y, r) # Nx3

            #Weighting: 
            weights = self._bi_square_weights(e)

            e_weighted = weights*e

            e_sorted_indices = np.argsort(e_weighted)
            sorted_pointcloud = pointcloud[e_sorted_indices]
            top_h_points = sorted_pointcloud[:h, :]

        return x, y, r

    def _bi_square_weights(self, residuals):
        """
        Tukey's well-known robust'bi-square' weight function
        """
        e_star = residuals / (6*np.median(np.abs(residuals)))
        # weights = np.zeros(e_star.shape)

        weights = np.where(e_star < 1, np.square(1-np.square(e_star)), 0)

        return weights

        
    def _compute_residuals(self, q, a0, b0, r0):
        """
        residual computation equation: 
        """
        e = np.sqrt(np.square(q[:, 0] - a0) - np.square(q[:, 1] - b0)) - r0
        return e



        # print( x, y, r, s)


    # def _get_length(self, )



# if __name__ == "__main__":
#     

#     # point_cloud = create_fake_cylinder(length=3, noise=False)
#     point_cloud = load_point_cloud('long_trunk.laz')
#     point_cloud = np.unique(point_cloud, axis = 0)
#     # point_cloud = generate_noisy_line()
#     point_cloud = cylinder_fitter.normalize_pointcloud(point_cloud)
#     first_pc, second_pc, third_pc = cylinder_fitter._get_pcs(point_cloud)        
#     print("Length", cylinder_fitter._get_length(point_cloud, first_pc))
#     plot_cylinder(point_cloud, first_pc, second_pc, third_pc)


if __name__ == "__main__":
    cylinder_fitter = RobustCylinderFitting()
    point_cloud = load_point_cloud('small_log.laz')
    point_cloud = np.unique(point_cloud, axis = 0)
    cylinder_fitter.fit(point_cloud)
    
        

    # first_pc, second_pc, third_pc = cylinder_fitter.get_cylinder_orientation(point_cloud)

    
