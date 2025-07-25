from robpy.covariance import DetMCD,FastMCD
import numpy as np
import math
from circle_fit import hyperSVD
from scipy.spatial.distance import pdist


class RobustCylinderFitter: 
    """   
    Approximate Implementation of 'Robust cylinder fitting in laser scanning point cloud data' 
    by Abdul Nurunnabi, Yukio Sadahiro, Yukio Sadahiro
    
    Robust Cylinder fitting class. 

    Usage: 
    >>> fitter = RobustCylinderFitting()
    >>> start, axis,  r, l = fitter.fit(point_cloud_of_cylinder)
    """
    def __init__(self):
        pass

    def fit(self, point_cloud):
        """
        Main process step

        Parameters:
            point_cloud: nx3 point cloud representing a cylinder in space 

        Return: 
            start: 1x3 array representing (X,Y,Z) of start of cylinder
            axis: 1x3 unit vector representing axis direction of cylinder
            r: (float) radius of cylinder
            l: (float) length of cylinder
        """
        # Task 1: get cylinder orientation. 
        point_cloud, data_mean = self._normalize_pointcloud(point_cloud)
        
        pc_data = self._get_pcs(point_cloud)
        if pc_data is None:
            return None
        else:
            pc1, pc2, pc3, mean = pc_data
        self.mcd_mean = mean
        l = self._get_length(point_cloud, pc1)
        
        # Task 2: Robust circle fitting
        circle_params = self._fit_circle(point_cloud, pc2, pc3, method="rlts")
        if circle_params is None: 
            print("Cylinder Fitter: Failed to Fit Cylinder")
            return None
        else:
            x, y, r, s = circle_params
            self.x = x
            self.y = y
            self.s = s

        # Task 3: Cylinder Start and axis
        center = self.get_cylinder_center(pc2, pc3, x, y, mean)
        axis = self.get_cylinder_axis(pc1)
        start = self.get_cylinder_start(pc2, pc3, center, axis, l)
        self.data_mean = data_mean
        # Return all data back to unnormalized.
        start = start + data_mean

        return start, axis,  r, l

    def get_cylinder_axis(self, pc1):
        """
        Get the unit vector representing the direction of the cylinder in space. 

        Return: (Nx3) unit vector.
        """
        unit_vector = pc1 / np.linalg.norm(pc1)
        return unit_vector

    def get_cylinder_start(self, pc2, pc3, center, axis, l):
        """
        Get start according to equation: 
        """
        start = center - (l/2) * axis
        return start

    def get_cylinder_center(self, pc2, pc3, x, y, mcd_mean):
        """
        PC2 (v1) and PC3 (vo)
        """
        center = pc2*x + pc3*y 
        return center

    def _get_pcs(self, point_cloud):
        """
        Task 1: Using MCD instead for cylinder orientation
        """
        mcd = FastMCD()
        try:
            covariance = mcd.calculate_covariance(point_cloud)
        except:
            try:
                covariance = DetMCD().calculate_covariance(point_cloud)
            except Exception as e:
                print("Cylinder Fitter: Failed to find covariance")
                return None
                

        covariance = mcd.calculate_covariance(point_cloud)
        mean = mcd.location_
        U, S, Vt = np.linalg.svd(covariance, full_matrices=False)
        first_pc = Vt[0, :] 
        second_pc = Vt[1, :] 
        third_pc = Vt[2, :] 
        return first_pc, second_pc, third_pc, mean


    def _normalize_pointcloud(self, point_cloud):
        """
        centers the pointcloud around 0,0,0 
        """
        mean = np.mean(point_cloud, axis=0)
        normalized = point_cloud - mean
        return normalized, mean
    
    def _get_length(self, cylinder, pc1):
        """
        Returns length of cylinder
        """

        projection = np.dot(cylinder, pc1)
        length = np.max(projection) - np.min(projection)
        return length
    
    def _fit_circle(self, point_cloud, pc2, pc3, method="wrlts"):
        """
        Fits a circle to the cylinder point cloud.
        """
        # Project points onto the orthogonal plane created by pc2, pc3
        projection1 = np.dot(point_cloud, pc2)
        projection2 = np.dot(point_cloud, pc3)
        circle_projection = np.array([projection1, projection2])
        self.circle_projection = np.transpose(circle_projection)
        if method == "wrlts":
            circle_params = self._WRLTS(self.circle_projection) # For Me WRLTS had better results on tree branches.
        elif method == "rlts":
            circle_params = self._RLTS(self.circle_projection)
        else:
            raise ValueError(f"Unknown method {method}")
        if circle_params is None: 
            return None
        else:
            x, y, r, s = circle_params
        return x, y, r, s
    
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
        x, y, r, s = hyperSVD(random_points)

        e = self._compute_residuals(point_cloud, x, y, r) # Nx3

        # Then we sort the points according to their residuals. 
        e_sorted_indices = np.argsort(e)
        sorted_pointcloud = point_cloud[e_sorted_indices]
        top_h_points = sorted_pointcloud[:h, :]

        # Then iterate to find better points. 
        for i in range(100):
            try:
                x, y, r, s = hyperSVD(top_h_points)
            except Exception as e:
                print("Cylinder Fitter: Circle Fitting Failed During Cylinder Fitting.")
                return None
            e = self._compute_residuals(point_cloud, x, y, r) # Nx3
            e_sorted_indices = np.argsort(e)
            sorted_pointcloud = point_cloud[e_sorted_indices]
            top_h_points = sorted_pointcloud[:h, :]

        return x, y, r, s

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
        x, y, r, s = hyperSVD(random_points)

        e = self._compute_residuals(point_cloud, x, y, r) # Nx3

        # Then we sort the points according to their residuals. 
        e_sorted_indices = np.argsort(e)
        sorted_pointcloud = point_cloud[e_sorted_indices]
        top_h_points = sorted_pointcloud[:h, :]

        # Then iterate to find better points.
        for i in range(100):
            try:
                x, y, r, s = hyperSVD(top_h_points)
            except Exception as e:
                print("Cylinder Fitter: Circle Fitting Failed During Cylinder Fitting.")
                return None
            e = self._compute_residuals(point_cloud, x, y, r) # Nx3

            #Weighting: 
            weights = self._bi_square_weights(e)

            e_weighted = weights*e

            e_sorted_indices = np.argsort(e_weighted)
            sorted_pointcloud = point_cloud[e_sorted_indices]
            top_h_points = sorted_pointcloud[:h, :]

        return x, y, r, s

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


class RobustCylinderFitterEcomodel(RobustCylinderFitter):
    """
    Extends Robust cylinder fitting to include plausibility regarding cylinder fitting for tree segments. 

    Returns None if a cylinder could not be created.
    """
    def __init__(self):
        super().__init__()

    def fit(self, point_cloud, method="method_1"):
        if method == "default":
            return self.default(point_cloud)
        elif method == "method_1":
            return self.method_1(point_cloud)
        elif method == "method_2":
            return self.method_2()

    def method_1(self, point_cloud):
        """
        Compare the max distance between two points on the circle project plane to the measure radius to see
        if the estimated radius is too large
        """
        cylinder_params = super().fit(point_cloud)
        if cylinder_params is None:
            return None
        else:
            start, axis, r, l = cylinder_params

        max_distance = np.max(pdist(self.circle_projection))
        scale_factor = 2
        if r*2 > scale_factor*max_distance: 
            r = max_distance/2
            start = start = self.mcd_mean - (l/2) * axis + self.data_mean
            return start, axis,  r, l
        else:
            return start, axis,  r, l

    def method_2(self, point_cloud):
        """
        Just checks the cylinder. 
        """
        cylinder_params = super().fit(point_cloud)
        if cylinder_params is None:
            return None
        else:
            start, axis, r, l = cylinder_params

        if r > 0.25: 
            print("Cylinder Fitter: Radius too big")
            return None
        else:
            return start, axis,  r, l 
        
    def default(self, point_cloud):
        """
        Normal fit algorithm.
        """
        return super().fit(point_cloud)