"""
Python adaptation and extension of TREEQSM:

% This file is part of TREEQSM.
%
% TREEQSM is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% TREEQSM is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with TREEQSM.  If not, see <http://www.gnu.org/licenses/>.

Version: 0.0.1
Date: 5 Mar 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

from numba import jit
import numpy as np
from collections import deque
import LeastSquaresFitting.LSF as LSF
from Utils.Utils import distances_to_line,distances_between_lines,growth_volume_correction,surface_coverage_filtering,surface_coverage2,verticalcat
import matplotlib.pyplot as plt
import Utils.Utils as Utils

#CylinderFitting:
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


def plot_cyl(segment_point_cloud):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(segment_point_cloud[:, 0], segment_point_cloud[:, 1], segment_point_cloud[:, 2], color='green')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Scatter Plot')
    fig.axes
    plt.axis('equal')
    plt.show()


def cylinders(P,cover,segment,inputs):
    """
    Fits cylinders to the branch-segments of the point cloud
    Reconstructs the surface and volume of branches of input tree with
    cylinders. Subdivides each segment to smaller regions to which cylinders
    are fitted in least squares sense. Returns the cylinder information and
    in addition the child-relation of the cylinders plus the cylinders in
    each segment.


    Args:
        P:          Point cloud, matrix
        cover:      Cover sets
        segment:    Segments
        inputs:     Input parameters of the reconstruction:
            MinCylRad   Minimum cylinder radius, used in the taper corrections
            ParentCor   Radius correction based on radius of the parent: radii in
                          a branch are usually smaller than the radius of the parent
                          cylinder in the parent branch
            TaperCor    Parabola taper correction of radii inside branches.
            GrowthVolCor  If 1, use growth volume correction
            GrowthVolFac  Growth volume correction factor

    Returns:
        cylinder  Structure array containing the following cylinder info:
            radius        Radii of the cylinders, vector
            length        Lengths of the cylinders, vector
            axis          Axes of the cylinders, matrix
            start         Starting points of the cylinders, matrix
            parent        Parents of the cylinders, vector
            extension     Extensions of the cylinders, vector
            branch        Branch of the cylinder
            BranchOrder   Branching order of the cylinder
            PositionInBranch    Position of the cylinder inside the branch
            mad           Mean absolute distances of points from the cylinder
                                    surface, vector
            SurfCov       Surface coverage measure, vector
            added         Added cylinders, logical vector
            UnModRadius   Unmodified radii

    """
    # Initialization of variables
    #print(segment)
    Segs = segment['segments']
    SPar = segment['ParentSegment'].astype(np.int64)
    SChi = segment['ChildSegment']
    NumOfSeg = len(Segs)
    n_init = max(2000, min(40 * NumOfSeg, 200000))
    c = 0  # Number of cylinders determined
    CChi = [[] for _ in range(n_init)]  # Children of cylinders
    CiS = [[] for _ in range(NumOfSeg)]  # Cylinders in each segment
    cylinder = {
        'radius': np.zeros(n_init, dtype=np.float32),
        'length': np.zeros(n_init, dtype=np.float32),
        'start': np.zeros((n_init, 3), dtype=np.float32),
        'axis': np.zeros((n_init, 3), dtype=np.float32),
        'parent': np.zeros(n_init, dtype=np.int32)-1,
        'extension': np.zeros(n_init, dtype=np.uint32),
        'added': np.zeros(n_init, dtype=bool),
        'UnmodRadius': np.zeros(n_init, dtype=np.float32),
        'branch': np.zeros(n_init, dtype=np.uint16),
        'SurfCov': np.zeros(n_init, dtype=np.float32),
        'mad': np.zeros(n_init, dtype=np.float32),
    }


    # Determine suitable order of segments (from trunk to the "youngest" child)
    ###### FY note: if a seg has no parent, it's assigned 0 in SPar. Here I'm excluding tree base (Seg[0])
    # to avoid generate overlapping cylinders with tree base's children
    bases = [i for i in range(NumOfSeg) if (SPar[i] == -1)]
    SegmentIndex = []
    for base in bases:  # BFS from base
        queue = deque([base])
        while queue:
            current = queue.popleft()
            SegmentIndex.append(current)
            children = [child for child in SChi[current]]  # Adjust if SChi is 1-based
            queue.extend(children)

    # Fit cylinders individually for each segment
    # Process each segment in order
    for si in SegmentIndex:
        # print(si)
        Seg = Segs[si]  # the current segment under analysis
        nl = len(Seg)  # number of cover set layers in the segment

        # Extract cover sets and points in the segment
        '''Sets = []  # the cover sets in the segment
        IndSets = []
        current = 0
        for layer in Seg:
            Sets.extend(layer)
            layer_len = len(layer)
            IndSets.append((current, current + layer_len - 1))
            current += layer_len'''
        Sets, IndSets = verticalcat(Seg)  # the cover sets indices and (start, end) indices in the segment

        ns = len(Sets)  # number of cover sets in the current segment
        #print(Sets)
        #print(IndSets)
        #print(cover['ball'])
        Points = np.concatenate([cover['ball'][s] for s in Sets])  # the points in the segments
        #Points = Points.ravel(order='F')  # to make the sequence same as matlab
        np_points = Points.shape[0]  # number of points in the segment

        # Determine indexes of points for faster definition of regions
        # Calculate BallSize and IndPoints
        BallSize = [cover['ball'][s].shape[0] for s in Sets]
        layer_sizes = []
        for j in range(nl):
            start, end = IndSets[j]
            layer_size = sum(BallSize[start:end + 1])
            layer_sizes.append(layer_size)
        cum_sizes = np.cumsum(layer_sizes)
        start_indices = np.concatenate(([0], cum_sizes[:-1]))
        end_indices = cum_sizes - 1
        IndPoints = np.column_stack((start_indices, end_indices))  # indexes for points in each layer of the segment

        # Reconstruct only large enough segments
        base_layer_points = end_indices[0] - start_indices[0] + 1  # number of points in the base
        #print(nl)
        #print("np_points ", np_points)
        #print("base_layer_pt ", base_layer_points)
        if nl <= 1 or np_points <= base_layer_points or ns <= 2 or np_points <= 20 or len(Seg[0])==0:
            continue

        # Cylinder fitting
        cyl, Reg = cylinder_fitting(P, Points, IndPoints, nl, si)
        #print(cyl)
        nc = np.size(cyl['radius'])

        # Search possible parent cylinder
        if nc > 0 and si > 0:
            PC, cyl, added = parent_cylinder(SPar, SChi, CiS, cylinder, cyl, si)
            nc = np.size(cyl['radius'])
        else:
            PC = []
            added = False
        cyl['radius0'] = cyl['radius'].copy()
        #print(cyl)

        # Modify cylinders
        if nc > 0:
            # Define parent cylinder
            parcyl = {
                'radius': cylinder['radius'][PC],
                'length': cylinder['length'][PC],
                'start': cylinder['start'][PC],
                'axis': cylinder['axis'][PC],
            }
            cyl = adjustments(cyl, parcyl, inputs, Reg)

        # Save the cylinders if valid
        # if at least one acceptable cylinder, then save them
        if nc > 0 and np.min(cyl['radius']) > 0:
            # If the parent cylinder exists, set the parent-child relations
            if np.size(PC) > 0:
                cylinder['parent'][c] = PC  # c: number of cylinders determined, start with 0
                if cylinder['extension'][PC] == c:
                    branch_id = cylinder['branch'][PC]
                    cylinder['branch'][c:c + nc] = branch_id
                    CiS[branch_id].extend(range(c, c + nc))
                else:
                    CChi[PC].append(c)
                    cylinder['branch'][c:c + nc] = si
                    CiS[si].extend(range(c, c + nc))
            else:
                cylinder['branch'][c:c + nc] = si
                CiS[si].extend(range(c, c + nc))

            # Assign cylinder properties
            cylinder['radius'][c:c + nc] = cyl['radius']
            cylinder['length'][c:c + nc] = cyl['length']
            cylinder['start'][c:c + nc] = cyl['start']
            cylinder['axis'][c:c + nc] = cyl['axis']
            cylinder['parent'][c + 1:c + nc] = range(c, c + nc - 1)
            cylinder['extension'][c:c + nc - 1] = range(c + 1, c + nc)
            cylinder['UnmodRadius'][c:c + nc] = cyl['radius0']
            cylinder['SurfCov'][c:c + nc] = cyl['SurfCov']
            cylinder['mad'][c:c + nc] = cyl['mad']
            if added:
                cylinder['added'][c] = True
            c += nc  # number of cylinders so far

    # Trim cylinders to actual count
    for key in cylinder:
        cylinder[key] = cylinder[key][:c]

    ## Define outputs
    # Skipped

    # Define branching order
    BOrd = np.zeros(c, dtype=np.uint8)
    for i in range(c):
        if cylinder['parent'][i] > -1:
            p = cylinder['parent'][i]
            if cylinder['extension'][p] == i:
                BOrd[i] = BOrd[p]
            else:
                BOrd[i] = BOrd[p] + 1
    cylinder['BranchOrder'] = BOrd
    # Define the cylinder position inside the branch
    PiB = np.ones(c, dtype=np.uint16)
    for i in range(NumOfSeg):
        cyls = CiS[i]
        if cyls:
            PiB[cyls] = np.arange(0, len(cyls))
    cylinder['PositionInBranch'] = PiB

    # Growth volume correction
    if inputs.get('GrowthVolCor', False) and c > 0:
        cylinder = growth_volume_correction(cylinder, inputs)

    return cylinder



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

        e = self._compute_residuals(pointcloud, x, y, r) # Nx3

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
            e = self._compute_residuals(pointcloud, x, y, r) # Nx3
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

        e = self._compute_residuals(pointcloud, x, y, r) # Nx3

        # Then we sort the points according to their residuals. 
        e_sorted_indices = np.argsort(e)
        sorted_pointcloud = pointcloud[e_sorted_indices]
        top_h_points = sorted_pointcloud[:h, :]

        # Then repeat the algorithm. 
        for i in range(100):
            x, y, r, s = hyperSVD(top_h_points)
            e = self._compute_residuals(pointcloud, x, y, r) # Nx3

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




# @jit(nopython=True, cache=True)
def cylinder_fitting(P, Points, Ind, nl, si):
    """
        Fits cylinders to regions of a point cloud segment.

        Parameters:
            P (np.ndarray):     The full point cloud (N x 3).
            Points (np.ndarray): Points in the segment.
            Ind (np.ndarray):   Indices of points in each layer of the segment.
            nl (int):           Number of layers in the segment.
            si (int):           Segment index.

        Returns:
            cyl (dict):     Cylinder parameters.
            Reg (list):     Regions used for fitting.

    """

    # Display the segment. 
    
    print('--------------------------------------')
    # print(P.shape)
    # print(Points.shape)
    # print(Ind.shape)
    # print(nl)
    # print(si)
    segment_point_cloud = P[Points]
    np.savetxt(f"segments/segment_{si}.xyz", segment_point_cloud)
    # cylinder_fitter = RobustCylinderFitting()   
    # x, y, r = cylinder_fitter.fit(segment_point_cloud)
    # print('x', x)
    # print('y', y)
    # print('r', r)
    # plot_cyl(segment_point_cloud)
    print('--------------------------------------')

    CylTop = np.zeros(3)  
    #print("nl ", nl)
    if nl > 6:
        i0 = 0  # Index of the first layer 1->0
        i = 3  # Index of the last layer 4->3
        t = 0  # Region counter
        Reg = [None] * nl  # Regions
        cyls = [None] * 11  # Cylinders
        regs = [None] * 11  # Regions for cylinders
        data = np.zeros((11, 4))  # Fit data

        while i0 < nl - 3:
            data = np.zeros((11, 4))
            # Fit at least three cylinders of different lengths
            bot = Points[Ind[i0, 0]:Ind[i0 + 1, 1] + 1]
            Bot = np.mean(P[bot],axis = 0)  # Bottom axis point of the region
            again = True
            j = 0
            c0 = {}
            while i + j <= nl-1 and j <= 10 and (j <= 2 or again):
                ## Select points and estimate axis
                RegC = Points[Ind[i0, 0]:Ind[i + j , 1] +1]  # Candidate region
                top = Points[Ind[i + j -1 , 0]:Ind[i + j, 1] + 1]  # Top axis point of the region
                Top = np.mean(P[top],axis = 0)
                # Axis of the cylinder:
                Axis = Top - Bot
                c0['axis'] = Axis / np.linalg.norm(Axis)  # normalized
                # Compute height along the axis
                h = (P[RegC] - Bot) @ c0['axis'].T
                minh = np.min(h)
                # Correct Bot to correspond to the real bottom
                if j == 0:
                    Bot = Bot + minh * c0['axis']
                    c0['start'] = Bot
                    h = (P[RegC] - Bot) @ c0['axis'].T
                    minh = np.min(h)
                if i + j >= nl-1:
                    ht = (Top - c0['start']) @ c0['axis'].T
                    Top = Top + (np.max(h) - ht) * c0['axis']
                #print(c0['start'])
                # Compute height of the Top
                ht = (Top - c0['start']) @ c0['axis'].T
                Sec = ((h-ht) <=.00001 ) & ((h-minh) >= -.00001)  # Only points below the Top
                c0['length'] = ht - minh  # Length of the region/cylinder
                # The region for the cylinder fitting:
                reg = RegC[Sec]
                Q0 = P[reg]

                ## Filter points and estimate radius
                #print(Q0, c0)
                if Q0.shape[0] > 20:
                    axis = c0['axis']
                    start = c0['start']
                    Keep, R_final,SurfCov,mad = surface_coverage_filtering(Q0, axis,start, c0['length'],0.02, 20)
                    c0['radius'] = R_final
                    c0['SurfCov'] = SurfCov
                    c0['mad'] = mad
                    c0['conv'] = 1
                    c0['rel'] = 1
                    reg = reg[Keep]
                    Q0 = Q0[Keep]
                else:
                    c0['radius'] = 0.01
                    c0['SurfCov'] = 0.05
                    c0['mad'] = 0.01
                    c0['conv'] = 1
                    c0['rel'] = 1

                ## Fit cylinder
                if Q0.shape[0] > 9:
                    if i >= nl-1 and t == 0:
                        c = LSF.least_squares_cylinder(Q0, c0)
                    elif i >= nl-1 and t > 0:
                        h = (Q0 - CylTop) @ c0['axis'].T
                        I = h >= 0
                        Q = Q0[I]  # The section
                        reg = reg[I]
                        n2 = Q.shape[0]
                        n1 = np.sum(~I)
                        if n2 > 9 and n1 > 5:
                            Q0 = np.vstack((Q0[~I], Q))  # the point cloud for cylinder fitting
                            W = np.hstack((1 / 3 * np.ones(n2), 2 / 3 * np.ones(n1)))  # the weights
                            c = LSF.least_squares_cylinder(Q0, c0, W, Q)
                        else:
                            c = LSF.least_squares_cylinder(Q0, c0)
                    elif t == 0:
                        top = Points[Ind[i + j - 3, 0]:Ind[i + j - 2, 1] + 1]
                        Top = np.mean(P[top], axis = 0)
                        ht = (Top - Bot) @ c0['axis'].T
                        h = (Q0 - Bot) @ c0['axis'].T
                        I = h <= ht
                        Q = Q0[I]  # the section
                        reg = reg[I]
                        n2 = Q.shape[0]
                        n3 = np.sum(~I)
                        if n2 > 9 and n3 > 5:
                            Q0 = np.vstack((Q, Q0[~I]))  # the point cloud for cylinder fitting
                            W = np.hstack((2 / 3 * np.ones(n2), 1 / 3 * np.ones(n3)))  # the weights
                            c = LSF.least_squares_cylinder(Q0, c0, W, Q)
                        else:
                            c = LSF.least_squares_cylinder(Q0, c0)
                    else:
                        top = Points[Ind[i + j - 3, 0]:Ind[i + j - 2, 1] + 1]
                        Top = np.mean(P[top], axis = 0)  # Top axis point of the region
                        ht = (Top - CylTop) @ c0['axis'].T
                        h = (Q0 - CylTop) @ c0['axis'].T
                        I1 = h < 0  # Bottom
                        I2 = (h >= 0) & (h <= ht)  # Section
                        I3 = h > ht  # Top
                        Q = Q0[I2]
                        reg = reg[I2]
                        n1 = np.sum(I1)
                        n2 = Q.shape[0]
                        n3 = np.sum(I3)
                        if n2 > 9:
                            Q0 = np.vstack((Q0[I1], Q, Q0[I3]))
                            W = np.hstack((1 / 4 * np.ones(n1), 2 / 4 * np.ones(n2), 1 / 4 * np.ones(n3)))
                            c = LSF.least_squares_cylinder(Q0, c0, W, Q)
                        else:
                            c = c0.copy()
                            c['rel'] = 0

                    if c['conv'] == 0:
                        c = c0.copy()
                        c['rel'] = 0
                    if (c['SurfCov']-.2) < -0.000001:
                        c['rel'] = 0
                else:
                    c = c0.copy()
                    c['rel'] = 0

                # Collect fit data
                data[j, :] = [c['rel'], c['conv'], c['SurfCov'], c['length'] / c['radius']]
                cyls[j] = c
                regs[j] = reg
                j += 1
                # If reasonable cylinder fitted, then stop fitting new ones
                # (but always fit at least three cylinders)
                RL = c['length'] / c['radius']
                if again and c['rel'] and c['conv'] and RL > 2:
                    if si == 1 and c['SurfCov'] > 0.7:
                        again = False
                    elif si > 1 and c['SurfCov'] > 0.5:
                        again = False

            ## Select the best of the fitted cylinders
            # based on maximum surface coverage
            ##changed from j+1->j
            OKfit = (data[:j, 0] !=0 ) & (data[:j, 1] != 0) & (data[:j, 3] > 1.5)
            J = np.arange(len(OKfit))
            if len(J) == 0:
                J = np.array([0])
            t += 1
            if np.any(OKfit):
                J = J[OKfit]
            I = np.argmax(data[J, 2] - 0.01 * data[J, 3])
            J = J[I]
            c = cyls[J]

            ## Update indices of the layers for the next region
            CylTop = c['start'] + c['length'] * c['axis']
            i0 += 1
            bot = Points[Ind[i0, 0]:Ind[i0 + 1, 1] + 1]
            Bot = np.mean(P[bot],axis = 0)  # Bottom axis point of the region
            h = (Bot - CylTop) @ c['axis'].T
            i00 = i0
            while i0 + 1 < nl-1 and i0 < i00 + 5 and h < -c['radius'] / 3:
                i0 += 1
                bot = Points[Ind[i0, 0]:Ind[i0+1, 1] + 1]
                Bot = np.mean(P[bot],axis = 0)
                h = (Bot - CylTop) @ c['axis'].T
            i = i0 + 5
            i = min(i, nl-1)

            ## If the next section is very short part of the end of the branch
            # then simply increase the length of the current cylinder
            if nl - i0 + 2 < 5:
                reg = Points[Ind[nl - 6, 0]:Ind[nl - 1, 1] + 1]
                Q0 = P[reg]
                ht = (c['start'] + c['length'] * c['axis']) @ c['axis'].T
                h = Q0 @ c['axis'].T
                maxh = np.max(h)
                if maxh > ht:
                    c['length'] += (maxh - ht)
                i0 = nl

            Reg[t - 1] = regs[J]

            if t == 1:
                cyl = c.copy()
                names = list(cyl.keys())
                for k in range(len(names)):
                    cyl[names[k]] = np.array([cyl[names[k]]])
                # cyl['axis']= np.array([cyl['axis']])
                # cyl['start'] = np.array([cyl['start']])
                
                
                n = len(names)
            else:
                for k in range(n):
                    if names[k] in ['axis','start']:
                        cyl[names[k]] = np.concatenate([cyl[names[k]], np.array([c[names[k]]])])
                    else:
                        cyl[names[k]] = np.append(cyl[names[k]], c[names[k]])

            ## Compute cylinder top for the definition of the next section
            CylTop = c['start'] + c['length'] * c['axis']

        Reg = Reg[:t]
        #print(c)

    else:
        ## Define a region for small segments
        Q0 = P[Points]
        #print(Points)
        if Q0.shape[0] > 10:
            ## Define the direction
            #print(Ind)  # Ind is the same
            bot = Points[Ind[0, 0]:Ind[0, 1] + 1]
            Bot = np.mean(P[bot],axis = 0)
            top = Points[Ind[nl - 1, 0]:Ind[nl - 1, 1] + 1]
            Top = np.mean(P[top],axis = 0)
            Axis = Top - Bot
            #print(Top)
            c0 = {'axis': Axis / np.linalg.norm(Axis)}
            h = Q0 @ c0['axis'].T
            c0['length'] = np.max(h) - np.min(h)
            hpoint = Bot @ c0['axis'].T
            c0['start'] = Bot - (hpoint - np.min(h)) * c0['axis']

            ## Define other outputs
            #print("Q0 ", Q0, "c0", c0)  # Q0 is the same
            axis = c0['axis']
            start = c0['start']
            Keep, R_final,SurfCov,mad = surface_coverage_filtering(Q0, axis,start,c0['length'], 0.02, 20)
            c0['radius'] = R_final
            c0['SurfCov'] = SurfCov
            c0['mad'] = mad
            c0['conv'] = 1
            c0['rel'] = 1
            #print("Keep ", Keep, "c0", c0)
            Reg = [Points[Keep]]
            Q0 = Q0[Keep]
            cyl = LSF.least_squares_cylinder(Q0, c0)
            if not cyl['conv'] or not cyl['rel']:
                cyl = c0
            t = 1
            names = list(cyl.keys())
            for k in range(len(names)):
                cyl[names[k]] = np.array([cyl[names[k]]])
        else:
            cyl = 0
            t = 0

    # Define Reg as coordinates
    for i in range(t):
        Reg[i] = P[Reg[i]]
    Reg = Reg[:t]

    return cyl, Reg

    '''
    return {'radius': np.array([]), 'length': np.array([]), 'start': np.array([]),
        'axis': np.array([]), 'UnmodRadius': np.array([]), 'SurfCov': np.array([]),
        'mad': np.array([])}, None
    '''


def parent_cylinder(SPar, SChi, CiS, cylinder, cyl, si):
    '''
    Finds the parent cylinder from the possible parent segment.
    Does this by checking if the axis of the cylinder, if continued, will
    cross the nearby cylinders in the parent segment.
    Adjust the cylinder so that it starts from the surface of its parent.

    Args:
        SPar:       Parent segment
        SChi:       Child segment
        CiS:        Cylinders in each segment
        cylinder:   Cylinder data structure
        cyl:        Cylinder parameters
        si:         Segment index

    Returns:
        PC:         Parent cylinder
        cyl:        Cylinder parameters
        added:      Added cylinders, logical vector

    '''
    
    # Extract current cylinder properties
    if np.size(cyl['radius']) > 1:
        rad = cyl['radius'].copy()
        len_cyl = cyl['length'].copy()
        sta = cyl['start'].copy()
        axe = cyl['axis'].copy()
    else:
        if type(cyl['radius']) != np.ndarray:
            rad = np.array([cyl['radius']])
            len_cyl = np.array([cyl['length']])

        else:
            rad = cyl['radius']
            len_cyl = cyl['length']
        if len(cyl['axis'].shape) == 2:
            sta = cyl['start']
            axe = cyl['axis']
        elif len(cyl['axis'].shape) == 1:
            sta = np.array([cyl['start']])
            axe = np.array([cyl['axis']])
        else:
            raise Exception("Too many axis dimensions")
    # PC     Parent cylinder
    nc = np.size(rad)
    added = False
    PC = np.array([], dtype=int)

    if SPar[si] > -1:  # Parent segment exists, find the parent cylinder
        s = SPar[si]
        parent_cylinders = CiS[s]  # the cylinders in the parent segment
        # select the closest cylinders for closer examination
        if len(parent_cylinders) > 1:
            # Calculate distances to potential parent cylinders
            pc_start = cylinder['start'][parent_cylinders]
            D = -pc_start+sta[0]#pc_start - sta[0]
            d = np.sum(D ** 2, axis=1)
            sorted_indices = np.argsort(d)

            if len(parent_cylinders) > 3:
                candidate_indices = sorted_indices[:4]
            else:
                candidate_indices = sorted_indices
            pc_candidates = np.array(parent_cylinders)[candidate_indices]
            ParentFound = False
        elif len(parent_cylinders) == 1:
            PC = np.array(parent_cylinders[0])
            ParentFound = True
        else:
            PC = np.array([], dtype=int)
            ParentFound = True

        ## Check possible crossing points
        if not ParentFound:
            # Calculate the possible crossing points of the cylinder axis, when
            # extended, on the surfaces of the parent candidate cylinders
            x = np.zeros((len(pc_candidates), 2))  # how much the starting point has to move to cross
            h = np.zeros((len(pc_candidates), 2))  # the crossing point height in the parent

            parent_axes = cylinder['axis'][pc_candidates]
            parent_starts = cylinder['start'][pc_candidates]
            parent_radii = cylinder['radius'][pc_candidates]

            for j in range(len(pc_candidates)):
                #  intersection points calculation solved from a quadratic equation
                a_dot = np.dot(axe[0], parent_axes[j].transpose())
                A = axe[0] - a_dot * parent_axes[j]
                B = sta[0] - parent_starts[j] - np.dot(sta[0], parent_axes[j]) * parent_axes[j] + \
                    np.dot(parent_starts[j], parent_axes[j]) * parent_axes[j]
                e = np.dot(A, A)
                f = 2 * np.dot(A, B)
                g = np.dot(B, B) - parent_radii[j] ** 2

                discriminant = f ** 2 - 4 * e * g
                if discriminant >= 0:
                    sqrt_disc = np.sqrt(discriminant)
                    s1 = (-f + sqrt_disc) / (2 * e)
                    s2 = (-f - sqrt_disc) / (2 * e)  # how much the starting point must be moved to cross
                    # The heights of the crossing points
                    x[j] = [s1, s2]
                    h[j, 0] = np.dot(sta[0], parent_axes[j]) + s1 * a_dot - np.dot(parent_starts[j], parent_axes[j])
                    h[j, 1] = np.dot(sta[0], parent_axes[j]) + s2 * a_dot - np.dot(parent_starts[j], parent_axes[j])

            ## Extend to crossing point in the (extended) parent
            # Filter valid crossing points (non-zero x1 values)
            valid_mask = x[:, 0] != 0
            pc = pc_candidates[valid_mask]
            x = x[valid_mask]
            h = h[valid_mask]

            # Initialize tracking variables
            j = 0
            n = len(pc)
            X = np.zeros((n, 3))  # [x_value, h_value, flag]
            parent_lengths = cylinder['length'][pc]
            ParentFound = False

            while j < n and not ParentFound:
                x1, x2 = x[j, 0], x[j, 1]
                h1, h2 = h[j, 0], h[j, 1]
                current_parent_length = parent_lengths[j]

                # Case 1: x1 positive, x2 negative
                # sp inside the parent and crosses its surface
                if x1 > 0 and x2 < 0:
                    if (0 <= h1 <= current_parent_length) and (len_cyl[0] - x1 > 0):
                        PC = pc[j]
                        sta[0] += x1 * axe[0]
                        len_cyl[0] -= x1
                        ParentFound = True
                    elif len_cyl[0] - x1 > 0:
                        X[j] = [x1, abs(h1) if h1 < 0 else h1 - current_parent_length, 0]
                    else:
                        X[j] = [x1, h1, 1]

                # Case 2: x1 negative, x2 positive
                # sp inside the parent and crosses its surface
                elif x1 < 0 and x2 > 0 and (len_cyl[0] - x2 > 0):
                    if (0 <= h2 <= current_parent_length) and (len_cyl[0] - x2 > 0):
                        PC = pc[j]
                        sta[0] += x2 * axe[0]
                        len_cyl[0] -= x2
                        ParentFound = True
                    elif len_cyl[0] - x2 > 0:
                        X[j] = [x2, abs(h2) if h2 < 0 else h2 - current_parent_length, 0]
                    else:
                        X[j] = [x2, h2, 1]

                # Case 3: Both negative, x2 < x1
                # sp outside the parent and crosses its surface when extended backwards
                elif x1 < 0 and x2 < 0 and x2 < x1 and (len_cyl[0] - x1 > 0):
                    if (0 <= h1 <= current_parent_length) and (len_cyl[0] - x1 > 0):
                        PC = pc[j]
                        sta[0] += x1 * axe[0]
                        len_cyl[0] -= x1
                        ParentFound = True
                    elif len_cyl[0] - x1 > 0:
                        X[j] = [x1, abs(h1) if h1 < 0 else h1 - current_parent_length, 0]
                    else:
                        X[j] = [x1, h1, 1]

                # Case 4: Both negative, x2 > x1
                # sp outside the parent and crosses its surface when extended backwards
                elif x1 < 0 and x2 < 0 and x2 > x1 and (len_cyl[0] - x2 > 0):
                    if (0 <= h2 <= current_parent_length) and (len_cyl[0] - x2 > 0):
                        PC = pc[j]
                        sta[0] += x2 * axe[0]
                        len_cyl[0] -= x2
                        ParentFound = True
                    elif len_cyl[0] - x2 > 0:
                        X[j] = [x2, abs(h2) if h2 < 0 else h2 - current_parent_length, 0]
                    else:
                        X[j] = [x2, h2, 1]

                # Case 5: Both positive, x2 < x1
                # sp outside the parent but crosses its surface when extended forward
                elif x1 > 0 and x2 > 0 and x2 < x1 and (len_cyl[0] - x1 > 0):
                    if (0 <= h1 <= current_parent_length) and (len_cyl[0] - x1 > 0):
                        PC = pc[j]
                        sta[0] += x1 * axe[0]
                        len_cyl[0] -= x1
                        ParentFound = True
                    elif len_cyl[0] - x1 > 0:
                        X[j] = [x1, abs(h1) if h1 < 0 else h1 - current_parent_length, 0]
                    else:
                        X[j] = [x1, h1, 1]

                # Case 6: Both positive, x2 > x1
                # sp outside the parent and crosses its surface when extended forward
                elif x1 > 0 and x2 > 0 and x2 > x1 and (len_cyl[0] - x2 > 0):
                    if (0 <= h2 <= current_parent_length) and (len_cyl[0] - x2 > 0):
                        PC = pc[j]
                        sta[0] += x2 * axe[0]
                        len_cyl[0] -= x2
                        ParentFound = True
                    elif len_cyl[0] - x2 > 0:
                        X[j] = [x2, abs(h2) if h2 < 0 else h2 - current_parent_length, 0]
                    else:
                        X[j] = [x2, h2, 1]

                j += 1

            if not ParentFound and n > 0:
                # Find the candidate with minimal h-value (second column in X)
                H_idx = np.argmin(X[:, 1])
                H_val = X[H_idx, 1]
                X_row = X[H_idx]

                if X_row[2] == 0 and H_val < 0.1 * parent_lengths[H_idx]:
                    # Case: Valid close crossing point found
                    PC = pc[H_idx]
                    sta[0] += X_row[0] * axe[0]
                    len_cyl[0] -= X_row[0]
                    ParentFound = True
                else:
                    PC = pc[H_idx]

                    # Handle cylinder removal/adjustment
                    if nc > 1 and X_row[0] <= rad[0] and abs(X_row[1]) <= 1.25 * cylinder['length'][PC]:
                        # Remove first cylinder and adjust second
                        S = sta[0] + X_row[0] * axe[0]
                        V = sta[1] + len_cyl[1] * axe[1] - S
                        len_cyl = np.concatenate([[np.linalg.norm(V)], len_cyl[2:]])
                        axe = np.vstack([V / np.linalg.norm(V), axe[2:]])
                        sta = np.vstack([S, sta[2:]])
                        rad = rad[1:]
                        cyl['mad'] = cyl['mad'][1:]
                        cyl['SurfCov'] = cyl['SurfCov'][1:]
                        nc -= 1
                        ParentFound = True
                    elif nc > 1:
                        # Remove first cylinder
                        sta = sta[1:]
                        len_cyl = len_cyl[1:]
                        axe = axe[1:]
                        rad = rad[1:]
                        cyl['mad'] = cyl['mad'][1:]
                        cyl['SurfCov'] = cyl['SurfCov'][1:]
                        nc -= 1
                    elif len(SChi[si])==0:#not SChi[si]:
                        # Remove entire cylinder
                        nc = 0
                        PC = np.array([], dtype=int)
                        ParentFound = True
                        rad = np.array([])
                    elif X_row[0] <= rad[0] and abs(X_row[1]) <= 1.5 * cylinder['length'][PC]:
                        # Adjust existing cylinder
                        sta[0] += X_row[0] * axe[0]
                        len_cyl[0] = abs(X_row[0])
                        ParentFound = True

            '''
            The parent is the cylinder in the parent segment whose axis
            line is the closest to the axis line of the first cylinder
            Or the parent cylinder is the one whose base, when connected
            to the first cylinder is the most parallel.
            '''
            if not ParentFound:
                # Fallback: Find closest line match
                # Add new cylinder
                pc = pc_candidates.copy()

                # Get line distances
                
                Dist, _, DistOnLines = distances_between_lines(
                    sta[0], axe[0],
                    cylinder['start'][pc],
                    cylinder['axis'][pc]
                )


                # Filter valid distance ranges
                valid_mask = (DistOnLines >= 0) & (DistOnLines <= cylinder['length'][pc])
                if not np.any(valid_mask):
                    valid_mask = (DistOnLines >= -0.2 * cylinder['length'][pc]) & (
                                DistOnLines <= 1.2 * cylinder['length'][pc])

                if np.any(valid_mask):
                    pc = pc[valid_mask]
                    Dist = Dist[valid_mask]
                    DistOnLines = DistOnLines[valid_mask]

                    # Find closest candidate
                    min_idx = np.argmin(Dist)
                    PC = pc[min_idx]
                    DistOnLine = DistOnLines[min_idx]
                    # Calculate connection geometry
                    Q = cylinder['start'][PC] + DistOnLine * cylinder['axis'][PC]
                    V = sta[0] - Q
                    L = np.linalg.norm(V)
                    V_dir = V / L
                    # Calculate angular correction
                    angle = np.arccos(np.dot(V_dir, cylinder['axis'][PC]))
                    h_val = np.sin(angle) * L
                    S = Q + (cylinder['radius'][PC] / h_val) * L * V_dir
                    new_length = (h_val - cylinder['radius'][PC]) / h_val * L

                    if new_length > 0.01 and new_length / len_cyl[0] > 0.2:
                        # Add new cylinder segment
                        nc += 1
                        sta = np.vstack([S, sta])
                        rad = np.concatenate(([rad[0]], rad))
                        #rad = np.vstack([rad[0], rad])
                        axe = np.vstack([V_dir, axe])
                        len_cyl = np.concatenate(([new_length], len_cyl))
                        # Update cylinder properties
                        for prop in ['mad', 'SurfCov', 'rel', 'conv']:
                            cyl[prop] = np.concatenate([[cyl[prop][0]], cyl[prop]])
                        added = True
                else:
                    # Fallback direction-based matching
                    V = -(cylinder['start'][pc] - sta[0])
                    L0 = np.linalg.norm(V, axis=1)
                    V_dir = V / L0[:, None]
                    A = np.dot(V_dir, axe[0])
                    max_idx = np.argmax(A)
                    L1 = L0[max_idx]
                    PC = pc[max_idx]
                    V = V_dir[max_idx]
                    # Calculate geometric parameters
                    #angle = np.arccos(A[max_idx])
                    angle = np.arccos(np.dot(V, cylinder['axis'][PC]))
                    h_val = np.sin(angle) * L1
                    S = cylinder['start'][PC] + (cylinder['radius'][PC] / h_val) * L1 * V
                    new_length = (h_val - cylinder['radius'][PC]) / h_val * L1

                    if new_length > 0.01 and new_length / len_cyl[0] > 0.2:
                        # Add new cylinder segment
                        nc += 1
                        sta = np.vstack([S, sta])
                        rad = np.concatenate([[rad[0]], rad])
                        axe = np.vstack([V, axe])
                        len_cyl = np.concatenate([[new_length], len_cyl])
                        # Update cylinder properties
                        for prop in ['mad', 'SurfCov', 'rel', 'conv']:
                            cyl[prop] = np.concatenate([[cyl[prop][0]], cyl[prop]])
                        added = True

    else:  # No parent segment exists
        PC = np.array([], dtype=int)

    # Define the output
    cyl['start'] = sta[:nc] if np.size(sta[0]) == 3 else sta
    cyl['axis'] = axe[:nc] if np.size(axe[0]) == 3 else axe
    cyl['radius'] = rad[:nc] if np.size(rad) > 1 else rad
    cyl['length'] = len_cyl[:nc] if np.size(len_cyl) > 1 else len_cyl
    cyl['mad'] = cyl['mad'][:nc] if np.size(cyl['mad']) > 1 else cyl['mad']
    cyl['SurfCov'] = cyl['SurfCov'][:nc] if np.size(cyl['SurfCov']) > 1 else cyl['SurfCov']
    cyl['conv'] = cyl['conv'][:nc] if np.size(cyl['conv']) > 1 else cyl['conv']
    cyl['rel'] = cyl['rel'][:nc] if np.size(cyl['rel']) > 1 else cyl['rel']

    return PC, cyl, added


def adjustments(cyl, parcyl, inputs, Regs):
    nc = np.size(cyl['radius'])
    Mod = np.zeros(nc, dtype=bool)  # cylinders modified
    SC = cyl['SurfCov'] #if nc > 1 else np.array([cyl['SurfCov']])
    #print(cyl['axis'])
    #print(cyl['radius'])

    ## Determine the maximum and the minimum radius
    # The maximum based on parent branch
    #print(parcyl)
    if parcyl['radius'].size > 0:
        MaxR = 0.95 * parcyl['radius']
        MaxR = np.maximum(MaxR, inputs['MinCylRad'])
    else:
        # use the maximum from the bottom cylinders
        a = min(3, nc)
        MaxR = 1.25 * np.max(cyl['radius'][:a] if nc > 1 else cyl['radius'])
        #print(cyl['radius'])
    if nc > 1:
        MinR = np.min(cyl['radius'][SC >= 0.7]) if np.any(SC >= 0.7) else None
        if not (MinR is None) and np.min(cyl['radius']) < MinR / 2:
            MinR = np.min(cyl['radius'][SC >= 0.4]) if np.any(SC >= 0.4) else None
        elif MinR is None:
            MinR = np.min(cyl['radius'][SC >= 0.4]) if np.any(SC >= 0.4) else None
            if MinR is None:
                MinR = inputs['MinCylRad']
    else:
        MinR = cyl['radius'] if SC >= 0.7 else None
        if MinR is None:
            MinR = cyl['radius'] if SC >= 0.4 else None
            if MinR is None:
                MinR = inputs['MinCylRad']
    #print(MaxR, MinR)

    ## Check maximum and minimum radii
    I = cyl['radius'] < MinR
    # if np.size(cyl['radius']) > 1:
    cyl['radius'][I] = MinR
    # elif I:
    #     cyl['radius'] = MinR
    Mod[I] = True

    if inputs['ParentCor'] or nc <= 3:
        #print(cyl)
        I = ((cyl['radius'] > MaxR) & (SC < 0.7)) | (cyl['radius'] > 1.2 * MaxR)
        # if np.size(cyl['radius']) > 1:
        cyl['radius'][I] = MaxR
        # elif I:
            # cyl['radius'] = MaxR
        Mod[I] = True
        # For short branches modify with more restrictions
        if nc <= 3:
            I = (cyl['radius'] > 0.75 * MaxR) & (SC < 0.7)
            if np.any(I):
                cyl_radius = cyl['radius'][I] if nc > 1 else cyl['radius']
                r = np.maximum(SC[I] / 0.7 * cyl_radius, MinR)
                # if np.size(cyl['radius']) > 1:
                cyl['radius'][I] = r
                # elif I:
                    # cyl['radius'] = r
                Mod[I] = True

    ##  Use taper correction to modify radius of too small and large cylinders
    # Adjust radii if a small SurfCov and high SurfCov in the previous and following cylinders
    for i in range(1, nc - 1):
        if SC[i] < 0.7 and SC[i - 1] >= 0.7 and SC[i + 1] >= 0.7:
            cyl['radius'][i] = 0.5 * (cyl['radius'][i - 1] + cyl['radius'][i + 1])
            Mod[i] = True

    if inputs['TaperCor']:
        if np.max(cyl['radius']) < 0.001:
            # Adjust radii of thin branches to be linearly decreasing
            if nc > 2:
                r = np.sort(cyl['radius'])[1:-1]
                a = 2 * np.mean(r)
                if a > max(r):
                    a = min(0.01, max(r))
                b = min(0.5 * np.min(cyl['radius']), 0.001)
                cyl['radius'] = np.linspace(a, b, nc)
            elif nc > 1:
                r = np.max(cyl['radius'])
                cyl['radius'] = np.array([r, 0.5 * r])
            Mod[:] = True

        ## Parabola adjustment of maximum and minimum
        # Define parabola taper shape as maximum (and minimum) radii for
        # the cylinders with low surface coverage
        elif nc > 4:
            branchlen = np.sum(cyl['length'])  # branch length
            L = cyl['length'] / 2 + np.concatenate([[0], np.cumsum(cyl['length'][:-1])])
            Taper = np.concatenate([L, [branchlen]])
            Taper_2 = np.concatenate([1.05*cyl["radius"],[MinR]])
            Taper = np.column_stack([Taper,Taper_2])
            sc = np.concatenate([SC, [1]])

            # Least square fitting of parabola to "Taper":
            A = np.array([
                [np.sum(sc * Taper[:, 0] ** 4), np.sum(sc * Taper[:, 0] ** 2)],
                [np.sum(sc * Taper[:, 0] ** 2), np.sum(sc)]
            ])
            y = np.array([
                np.sum(sc * Taper[:, 1] * Taper[:, 0] ** 2),
                np.sum(sc * Taper[:, 1])
            ])
            # x = np.linalg.lstsq(A, y)[0]
            x = np.linalg.solve(A, y)
            x[0] = min(x[0], -0.0001)  # tapering from the base to the tip
            Ru = x[0] * L ** 2 + x[1]  # upper bound parabola
            #Ru = np.clip(Ru, MinR, MaxR)
            Ru[Ru < MinR] = MinR
            if np.max(Ru) > MaxR:
                a = np.max(Ru)
                Ru = MaxR / a * Ru
            Rl = 0.75 * Ru  # lower bound parabola
            Rl[Rl < MinR] = MinR

            # Modify radii based on parabola:
            # change values larger than the parabola-values when SC < 70%:
            I = (cyl['radius'] > Ru) & (SC < 0.7)
            cyl['radius'][I] = Ru[I] + (cyl['radius'][I] - Ru[I]) * SC[I] / 0.7
            Mod[I] = True
            # change values larger than the parabola-values when SC > 70% and
            # radius is over 33% larger than the parabola-value:
            I = (cyl['radius'] > 1.333 * Ru) & (SC >= 0.7)
            cyl['radius'][I] = Ru[I] + (cyl['radius'][I] - Ru[I]) * SC[I]
            Mod[I] = True
            # change values smaller than the downscaled parabola-values:
            I = (cyl['radius'] < Rl) & (SC < 0.7) | (cyl['radius'] < 0.5 * Rl)
            cyl['radius'][I] = Rl[I]
            Mod[I] = True
        ## Adjust radii of short branches to be linearly decreasing
        else:
            if np.sum(SC >= 0.7) > 1:
                a = np.max(cyl['radius'][SC >= 0.7])
                b = np.min(cyl['radius'][SC >= 0.7])
            elif np.sum(SC >= 0.7) == 1:
                a = np.max(cyl['radius'][SC >= 0.7])
                b = np.min(cyl['radius'])
            else:
                a = np.sum(cyl['radius'] * SC / np.sum(SC))
                b = np.min(cyl['radius'])
            Ru = np.linspace(a, b, nc)
            I = (SC < 0.7) & ~Mod
            cyl['radius'][I] = Ru[I] + (cyl['radius'][I] - Ru[I]) * SC[I] / 0.7
            Mod[I] = True

    ## Modify starting points by optimising them for given radius and axis
    nr = len(Regs)
    for i in range(nc):
        if Mod[i]:
            # Get relevant region points
            if nr == nc:
                Reg = Regs[i]
            elif i > 0:  # 0-based index
                Reg = Regs[i - 1]
            if nc >1: #np.size(cyl['start'][0]) == 3:
                cyl_start = cyl['start'][i]
                cyl_axis = cyl['axis'][i]
                cyl_mad = cyl['mad'][i]
                cyl_length = cyl['length'][i]
                cyl_radius = cyl['radius'][i]
            else:
                cyl_start = cyl['start'][0]
                cyl_axis = cyl['axis'][0]
                cyl_mad = cyl['mad']
                cyl_length = cyl['length'][0]
                cyl_radius = cyl['radius']
            #print(cyl_radius)
            radius = cyl['radius'][i] if np.size(cyl['radius']) > 1 else cyl['radius']
            radius0 = cyl['radius0'][i] if np.size(cyl['radius0']) > 1 else cyl['radius0']
            if abs(radius - radius0) > 0.005 and (nr == nc or (nr < nc and i > 0)):
                # Transform points to local coordinate system
                P = Reg - cyl_start
                U, V = Utils.orthonormal_vectors(cyl_axis)
                P_local = P @ np.column_stack([U, V])

                # Fit circle to points
                cir = LSF.least_squares_circle_centre(P_local, [0, 0], cyl['radius'][i])

                if cir['conv'] and cir['rel']:
                    # Adjust start position
                    cyl_start += cir['point'][0] * U + cir['point'][1] * V.T
                    cyl_mad = cir['mad']


                    # Recalculate distances
                    _, V_dist, h, _ = Utils.distances_to_line(Reg, cyl_axis, cyl_start)

                    # Adjust cylinder length if needed
                    if np.min(h) < -0.001:
                        cyl_length = np.max(h) - np.min(h)
                        cyl_start += np.min(h) * cyl_axis
                        _, V_dist, h, _ = Utils.distances_to_line(Reg, cyl_axis, cyl_start)

                    # Calculate surface coverage
                    a = max(0.02, 0.2 * cyl_radius)
                    nl = max(4, int(np.ceil(cyl_length / a)))
                    ns = max(10, min(36, int(np.ceil(2 * np.pi * cyl_radius / a))))
                    #print(a, nl, ns)
                    if np.size(cyl['SurfCov']) > 1:
                        cyl['SurfCov'][i] = surface_coverage2(
                            cyl_axis, cyl_length, V_dist, h, nl, ns
                        )
                    else:
                        cyl['SurfCov'] = surface_coverage2(
                            cyl_axis, cyl_length, V_dist, h, nl, ns
                        )
            cyl['start'][i] = cyl_start
            cyl['mad'][i] = cyl_mad
            cyl['length'][i] = cyl_length
            cyl['axis'][i] = cyl_axis
        

    ## Continuous branches adjustment
    # Make cylinders properly "continuous" by moving the starting points
    # Move the starting point to the plane defined by parent cylinder's top
    if nc > 1:
        for j in range(1, nc):  # 0-based index
            prev_end = cyl['start'][j - 1] + cyl['length'][j - 1] * cyl['axis'][j - 1]
            U = cyl['start'][j] - prev_end

            if np.linalg.norm(U) > 1e-4:
                # First define vector V and W which are orthogonal to the cylinder axis N
                N = cyl['axis'][j]
                if np.linalg.norm(N) > 1e-6:
                    V, W = Utils.orthonormal_vectors(N)
                    # Now define the new starting point, Solve linear system
                    A = np.column_stack([N, V, W])
                    x = np.linalg.lstsq(A, U.T)[0]
                    # Adjust start position and length
                    cyl['start'][j] -= x[0] * N
                    if x[0] > 0 or (cyl['length'][j] + x[0] > 0):
                        cyl['length'][j] += x[0]

    # Connect far away first cylinder to the parent
    #print(parcyl['radius'])
    if parcyl['radius'].size > 0:
        # Calculate distance to parent axis
        d, V_dir, h, B = distances_to_line(
            cyl['start'][0],
            parcyl['axis'],
            parcyl['start']
        )
        d -= parcyl['radius']

        if d > 0.001:
            # Calculate connection parameters
            taper_end = cyl['start'][0] + cyl['length'][0] * cyl['axis'][0]
            V_corr = parcyl['radius'] * V_dir / np.linalg.norm(V_dir)

            # Determine connection point
            if 0 <= h <= parcyl['length']:
                new_start = parcyl['start'] + B + V_corr
            elif h < 0:
                new_start = parcyl['start'] + V_corr
            else:
                new_start = parcyl['start'] + parcyl['length'] * parcyl['axis'] + V_corr

            # Update cylinder parameters
            cyl['axis'][0] = taper_end - new_start
            cyl['length'][0] = np.linalg.norm(cyl['axis'][0])
            cyl['axis'][0] /= cyl['length'][0]
            cyl['start'][0] = new_start

    return cyl