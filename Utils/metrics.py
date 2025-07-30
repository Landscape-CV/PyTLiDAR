"""
Gets the metrics of the point cloud. 
"""
import numpy as np

class Metrics:
    def __init__(self):
        self.total_mean_accuracy = 0
        self.total_mean_points_processed = 0
        self.total_mean_square_error = 0 
        self.total_mse_points_processed = 0
        self.cylinder_fit_accuracy = 0
        self.total_branches = 0
        self.total_fit_branches = 0


    def update_accuracy(self, points, start, axis, radius):
        """
        Mean distance from points to surface of cylinder
        
        Args:
            points: point cloud of branch 
            axis: Axis of estimated cylinder
            radius: radius of estimated cylinder
        """
        # Get point distance orthogonal to axis direction, 
        cross_prod = np.cross(points - start, axis)
        distance_to_axis = np.linalg.norm(cross_prod, axis=1) / np.linalg.norm(axis)
        distance_to_surface = distance_to_axis - radius
        num_points = np.size(points, axis=0)
        mean_distance_to_surface = np.mean(np.abs(distance_to_surface))
        self.total_mean_accuracy = (self.total_mean_accuracy * self.total_mean_points_processed \
                                    + mean_distance_to_surface * num_points) \
                                    / (num_points + self.total_mean_points_processed)
        self.total_mean_points_processed +=num_points
        
    def update_mean_squared_error(self, points, start, axis, radius):
        """
        Mean squared error from point to surface of cylinder. 
        """
        cross_prod = np.cross(points - start, axis)
        distance_to_axis = np.linalg.norm(cross_prod, axis=0) / np.linalg.norm(axis)
        distance_to_surface = distance_to_axis - radius
        num_points = np.size(points, axis=0)
        mse_distance = np.square(distance_to_surface)
        mean_mse_distance_to_surface = np.mean(mse_distance)
        self.total_mean_square_error = (self.total_mean_square_error * self.total_mse_points_processed \
                                    + mean_mse_distance_to_surface * num_points) \
                                    / (num_points + self.total_mse_points_processed)
        self.total_mse_points_processed +=num_points

    def increment_branch_fit_accuracy(self, successful):
        """
        Updates how many branches had a cylinder fit to them...
        """
        if successful:
            self.total_branches += 1
            self.total_fit_branches += 1 
        else:
            self.total_branches += 1

    def print_metrics(self):
        """
        Prints metrics for the resulting tile ecomodel.
        """
        print("-----------------------------")
        print(f"{'Accuracy:':<20}{self.total_mean_accuracy:>10.4f}")
        print(f"{'MSE:':<20}{self.total_mean_square_error:>10.4f}")
        if self.total_branches != 0:
            print(f"{'Branch Fit Accuracy:':<20}{(self.total_fit_branches / self.total_branches):>10.4f}")
        print("-----------------------------")
        
if __name__ == "__main__":
    metrics = Metrics()

    metrics.update_accuracy(np.array([[3,0,0], [2,0,0]]), start=np.array([0,0,0]), axis=np.array([0,0,1]), radius=1)
    metrics.print_metrics()

