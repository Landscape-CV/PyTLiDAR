"""
Approximate Implementation of 'Robust cylinder fitting in laser scanning point cloud data' 
by Abdul Nurunnabi, Yukio Sadahiro, Yukio Sadahiro
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



class RobustCylinderFitting: 
    def __init__(self):
        pass

    def fit(self, point_cloud, num, fig):
        """
        Main process step
        """
        # Task 1: get cylinder orientation. 
        point_cloud = self._normalize_pointcloud(point_cloud)
        pc1, pc2, pc3 = self._get_pcs(point_cloud)
        if pc1 is None or pc2 is None or pc3 is None: 
            x = 0
            y = 0
            r = 0
            circle_projection = np.array([])
        else:

            length = self._get_length(point_cloud, pc1)
            
            # Task 2: Robust circle fitting
            x, y, r, circle_projection = self._fit_circle(point_cloud, pc2, pc3)

            # Plot: 
        

        row = num % 3
        left_pos = row * 2 + 1
        right_pos = row * 2 + 2

        if circle_projection.any():
            ax1 = fig.add_subplot(3, 2, left_pos)
            ax1.scatter(circle_projection[:, 0], circle_projection[:, 1], s=1)
            ax1.axis('equal')
            circle2 = Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=1)
            ax1.add_patch(circle2)
            ax1.text(0.01, 0.99, f"Estimated Diameter: {r*2:0.5f} m",
                            transform=ax1.transAxes,  # Use axes coordinates
                                ha='left', va='top',     # Align text to top-left
                                fontsize=12, color='purple')

        ax2 = fig.add_subplot(3, 2, right_pos, projection='3d')
        
        ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], color='brown')
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Z axis')
        ax2.set_title('Branch Segment')
        plt.tight_layout()
        plt.axis('equal')
        # plt.show()



        return x, y, r, fig

    def _get_pcs(self, point_cloud):
        """
        Task 1: Using MCD instead for cylinder orientation
        """
        mcd = DetMCD()
        try:
            covariance = mcd.calculate_covariance(point_cloud)
        except np.linalg.LinAlgError:
            print("UNABLE TO GET BRANCH")
            return None, None, None
        U, S, Vt = np.linalg.svd(covariance, full_matrices=False)
        first_pc = Vt[0, :] 
        second_pc = Vt[1, :] 
        third_pc = Vt[2, :] 
        return first_pc, second_pc, third_pc


    def _normalize_pointcloud(self, point_cloud):
        """
        centers the pointcloud around 0,0,0 
        """
        mean = np.mean(point_cloud, axis=0)
        normalized = point_cloud - mean
        return normalized
    
    def _get_length(self, cylinder, pc1):
        """
        Returns length of cylinder
        """

        projection = np.dot(cylinder, pc1)
        length = np.max(projection) - np.min(projection)
        return length
    
    def _fit_circle(self, point_cloud, pc2, pc3):
        """
        Fits a circle to the cylinder point cloud.
        """
        # Project points onto the orthogonal plane created by pc2, pc3
        projection1 = np.dot(point_cloud, pc2)
        projection2 = np.dot(point_cloud, pc3)
        circle_projection = np.array([projection1, projection2])
        circle_projection = np.transpose(circle_projection)
        x, y, r = self._WRLTS(circle_projection) # For Me WRLTS had better results on tree branches.        
        return x, y, r, circle_projection
    
    def _RLTS(self, point_cloud):
        """
        Repeated Least Trimmed Square (RLTS)
        """
        h_0 = 4
        p_r = 0.999
        eps = 0.5
        h = math.ceil(point_cloud.shape[0] * 0.5)

        # Fit initial circle from randomly selected points
        indices = np.random.choice(point_cloud.shape[0], size=h_0, replace=False)
        random_points = point_cloud[indices]

        # Get circle parameters for initial guess
        print(random_points.shape)
        x, y, r, s = hyperSVD(random_points)

        e = self._compute_residuals(point_cloud, x, y, r) # Nx3

        # Then we sort the points according to their residuals. 
        e_sorted_indices = np.argsort(e)
        sorted_pointcloud = point_cloud[e_sorted_indices]
        top_h_points = sorted_pointcloud[:h, :]

        # Then iterate to find better points. 
        for i in range(100):
            x, y, r, s = hyperSVD(top_h_points)
            e = self._compute_residuals(point_cloud, x, y, r) # Nx3
            e_sorted_indices = np.argsort(e)
            sorted_pointcloud = point_cloud[e_sorted_indices]
            top_h_points = sorted_pointcloud[:h, :]

        return x, y, r

    def _WRLTS(self, point_cloud):
        """
        Weighted Repeated Least Trimmed Square (WRLTS)
        """
        h_0 = 4
        p_r = 0.999
        eps = 0.5
        h = math.ceil(point_cloud.shape[0] * 0.5)

        # Fit initial circle from randomly selected points
        indices = np.random.choice(point_cloud.shape[0], size=h_0, replace=False)
        random_points = point_cloud[indices]

        # Get circle parameters for initial guess
        print(random_points.shape)
        x, y, r, s = hyperSVD(random_points)

        e = self._compute_residuals(point_cloud, x, y, r) # Nx3

        # Then we sort the points according to their residuals. 
        e_sorted_indices = np.argsort(e)
        sorted_pointcloud = point_cloud[e_sorted_indices]
        top_h_points = sorted_pointcloud[:h, :]

        # Then iterate to find better points.
        for i in range(100):
            x, y, r, s = hyperSVD(top_h_points)
            e = self._compute_residuals(point_cloud, x, y, r) # Nx3

            #Weighting: 
            weights = self._bi_square_weights(e)

            e_weighted = weights*e

            e_sorted_indices = np.argsort(e_weighted)
            sorted_pointcloud = point_cloud[e_sorted_indices]
            top_h_points = sorted_pointcloud[:h, :]

        return x, y, r

    def _bi_square_weights(self, residuals):
        """
        Tukey's well-known robust'bi-square' weight function
        """
        e_star = residuals / (6*np.median(np.abs(residuals)))

        weights = np.where(e_star < 1, np.square(1-np.square(e_star)), 0)

        return weights

        
    def _compute_residuals(self, q, a0, b0, r0):
        """
        residual computation equation: 
        """
        e = np.sqrt(np.square(q[:, 0] - a0) - np.square(q[:, 1] - b0)) - r0
        return e
    
