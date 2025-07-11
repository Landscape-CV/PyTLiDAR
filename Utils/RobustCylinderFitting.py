from robpy.covariance import DetMCD
import numpy as np
import math
from circle_fit import hyperSVD


class RobustCylinderFitter: 
    """   
    Approximate Implementation of 'Robust cylinder fitting in laser scanning point cloud data' 
    by Abdul Nurunnabi, Yukio Sadahiro, Yukio Sadahiro
    
    Robust Cylinder fitting class. 

    Usage: 
    >>> fitter = RobustCylinderFitting()
    >>> X, Y, Z, r, l = fitter.fit(point_cloud_of_cylinder)
    """
    def __init__(self):
        pass

    def fit(self, point_cloud):
        """
        Main process step

        Parameters:
            point_cloud: nx3 point cloud representing a cylinder in space 

        Returns: 
            start: 1x3 array representing (X,Y,Z) of start of cylinder
            axis: 1x3 array representing axis of cylinder
            r: (float) radius of cylinder
            l: (float) length of cylinder
        """
        # Task 1: get cylinder orientation. 
        # point_cloud = self._normalize_pointcloud(point_cloud)
        pc1, pc2, pc3 = self._get_pcs(point_cloud)
        l = self._get_length(point_cloud, pc1)
        
        # Task 2: Robust circle fitting
        x, y, r = self._fit_circle(point_cloud, pc2, pc3)

        # Task 3: Cylinder Start and axis
        X, Y, Z = self.get_cylinder_start(pc1, pc2, x, y)
        axis = self.get_cylinder_axis(pc1)
        return (X, Y, Z), axis,  r, l

    def get_cylinder_axis(self, pc1):
        """
        Get the unit vector representing the direction of the cylinder in space. 

        Return: (Nx3) unit vector.
        """
        unit_vector = pc1 / np.linalg.norm(pc1)
        return unit_vector

    def get_cylinder_start(self, pc1, pc2, x, y):
        """
        Get start according to equation: 

        a*v1 + b*v1
        """
        start = pc1*x + pc2*y
        X = start[0]
        Y = start[1]
        Z = start[2]
        return X, Y, Z



    def _get_pcs(self, point_cloud):
        """
        Task 1: Using MCD instead for cylinder orientation
        """
        mcd = DetMCD()
        covariance = mcd.calculate_covariance(point_cloud)
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
        return x, y, r
    
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
    