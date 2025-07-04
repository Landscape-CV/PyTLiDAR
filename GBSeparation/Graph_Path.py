# Copyright (c) 2017-2019, Matheus Boni Vicari, TLSeparation Project
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2017-2019, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.3.2"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
def array_to_graph(arr, base_id, kpairs=3, knn=300, nbrs_threshold=0.1,
                   nbrs_threshold_step=0.02, graph_threshold=np.inf):

    """
    Converts a numpy.array of points coordinates into a Weighted BiDirectional
    NetworkX Graph.
    This funcions uses a NearestNeighbor search to determine points adajency.
    The NNsearch results are used to select pairs of points (or nodes) that
    have a common edge.

    Parameters
    ----------
    arr : array
        n-dimensional array of points.
    base_id : int
        Index of base id (root) in the graph.
    kpairs : int
        Number of points around each point in arr to select in order to
        build edges.
    knn : int
        Number of neighbors to search around each point in the neighborhood
        phase. The higher the better (careful, it's  memory intensive).
    nbrs_threshold : float
        Initial distance used in the final phase of edges generation.
    nbrs_threshold_step : float
        Distance increment used in the final phase of edges generation. It's
        used to make sure that in the end, every point in arr will be
        translated to nodes in the graph.
    graph_threshold : float
        Maximum distance between pairs of nodes (edge distance) accepted in
        the graph generation.

    Returns
    -------
    G : networkx graph
        Graph containing all points in 'arr' as nodes.

    """

    # Initializing graph.
    G = nx.Graph()

    # Generating array of all indices from 'arr' and all indices to process
    # 'idx'.
    idx_base = np.arange(arr.shape[0], dtype=int)

    # Initializing NearestNeighbors search and searching for all 'knn'
    # neighboring points arround each point in 'arr'.
    """
    JOHN'S NOTE:
        This will require the most updates. The way this currently works is to use the sklearn nearest neighbors up front and then still 
        iterate through the neighbors to add them to the graph.
        Instead, we can create a KDTree then iterate through the points to calculate the neighbors and add them to the graph in a single while loop.

        To see an example of this, see add_layer in cluster.py. 
        There, we make two graphs simultaneously for separate purposes, so focus on the full_pcd_tree and graph_points lines.
        The basic steps are:
        1. Create a KDTree from the points. (In cluster.py, this is done like so: 
            full_pcd_tree = cKDTree(full_pcd[:, :3], leafsize=15)pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            pcd_tree = o3d.geometry.KDTreeFlann(pcd)
            
            The main benefit of using open3d over sklearn is that it has the method search_hybrid_vector_3d which 
            allows us to limit the search range while we get the neighbors, instead of removing the neighbors out of range as is done the initial implementation.
        2. Initialize the graph vertices with all the points in the array. It is siginificantly faster to add all the points at once than to add them one by one.
        3. Iterate through the points and use search_hybrid_vector_3d to get the neighbors.
        4. For each neighbor, add edges to a list/array
           You may want to utilize the method make_edges from cluster.py, as it is optimized with numba so will be quite fast.
        5. After iterating through all the points, add the edges to the graph in a single call. Igraph is optimized for buk edge addition
        
        You may try implementing this exactly here, or you may want to incorporate some of the logic from the current implementation using this method.
        The main behavior worth keeping from the current implementation is the way it handles the nbrs_threshold and nbrs_threshold_step, i.e.
        continously increasing the threshold on unconnected nodes until the graph is fully connected.
    
    
    """
    nbrs = NearestNeighbors(n_neighbors=min(knn,len(arr)), metric='euclidean',
                            leaf_size=15, n_jobs=-1).fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    indices = indices.astype(int)

    # Connecting root point to it's knn.
    add_nodes(G, base_id, indices[-1], distances[-1], graph_threshold)

    # Initializing variables for current ids being processed (current_idx)
    # and all ids already processed (processed_idx).
    current_idx = indices[-1]
    processed_idx = indices[-1]
    unprocessed_idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]

    # Initializing temp_nbrs_threshold which will be changed in loop.
    temp_nbrs_threshold = nbrs_threshold

    # Looping while there are still indices (idx) left to process.

    while unprocessed_idx.shape[0] > 0:

        # If current_idx is a list containing several indices.
        if len(current_idx) > 0:

            # Selecting NearestNeighbors indices and distances for current
            # indices being processed.
            nn = indices[current_idx]
            dd = distances[current_idx]

            # Masking out indices already contained in processed_idx.
            mask1 = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)

            # Initializing temporary list of nearest neighbors. This list
            # is latter used to accumulate points that will be added to
            # processed points list.
            nntemp = []

            # Looping over current indices's set of nn points and selecting
            # knn points that hasn't been added/processed yet (mask1).
            for i, (n, d, g) in enumerate(zip(nn, dd, current_idx)):
                nn_idx = n[mask1[i]][0:kpairs+1]
                dd_idx = d[mask1[i]][0:kpairs+1]
                nntemp.append(nn_idx)

                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, graph_threshold)

            # Obtaining an unique array of points currently being processed.
            current_idx = np.unique([t2 for t1 in nntemp for t2 in t1])

            # reset the temp_nbrs_threshold
            temp_nbrs_threshold = nbrs_threshold

        # If current_idx is an empty list.
        elif len(current_idx) == 0:

            # Getting NearestNeighbors indices and distance for all indices
            # that remain to be processed.
            unprocessed_nn = indices[unprocessed_idx]
            unprocessed_dd = distances[unprocessed_idx]
            
            # Masking indices in idx2 that have already been processed. The
            # idea is to connect remaining points to existing graph nodes.
            mask1 = np.in1d(unprocessed_nn, processed_idx).reshape(unprocessed_nn.shape)
            if np.sum(mask1) == 0:
                break
            # Masking neighboring points that are withing threshold distance.
            mask2 = unprocessed_dd < temp_nbrs_threshold
            # mask1 AND mask2. This will mask only indices that are part of
            # the graph and within threshold distance.
            mask = np.logical_and(mask1, mask2)

            # Getting unique array of indices that match the criteria from
            # mask1 and mask2.
            temp_idx = np.unique(np.where(mask)[0])
            # Assigns remaining indices (idx) matched in temp_idx to
            # current_idx.
            current_idx = unprocessed_idx[temp_idx]

            # Selecting NearestNeighbors indices and distances for current
            # indices being processed.
            nn = indices[current_idx]
            dd = distances[current_idx]

            # Masking points in nn that have already been processed.
            # This is the oposite approach as above, where points that are
            # still not in the graph are desired. Now, to make sure the
            # continuity of the graph is kept, join current remaining indices
            # to indices already in G.
            mask = np.in1d(nn, processed_idx).reshape(nn.shape)

            # Looping over current indices's set of nn points and selecting
            # knn points that have alreay been added/processed (mask).
            # Also, to ensure continuity over next iteration, select another
            # kpairs points from indices that haven't been processed (~mask).
            for i, (n, d, g) in enumerate(zip(nn, dd, current_idx)):
                nn_idx = n[mask[i]][0:kpairs+1]
                dd_idx = d[mask[i]][0:kpairs+1]

                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, graph_threshold)

            # Check if current_idx is still empty. If so, increase the
            # nbrs_threshold to try to include more points in the next
            # iteration.
            if len(current_idx) == 0:
                temp_nbrs_threshold += nbrs_threshold_step

        # Appending current_idx to processed_idx.
        processed_idx = np.append(processed_idx, current_idx)

        # Generating list of remaining proints to process.
        unprocessed_idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]

        # print("unprocessed_count:", unprocessed_idx.shape[0])

    return G



def extract_path_info(G, base_id, return_path=False):

    """
    Extracts shortest path information from a NetworkX graph.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object from which to extract the information.
    base_id : int
        Base (root) node id to calculate the shortest path for all other
        nodes.
    return_path : boolean
        Option to select if function should output path list for every node
        in G to base_id.

    Returns
    -------
    nodes_ids : list
        Indices of all nodes in graph G.
    distance : list
        Shortest path distance (accumulated) from all nodes in G to base_id
        node.
    path_list : dict
        Dictionary of nodes that comprises the path of every node in G to
        base_id node.

    """

    # Checking if the function should also return the paths of each node and
    # if so, generating the path list and returning it.
    if return_path is True:
        path_dis, path_list = nx.single_source_dijkstra(G, base_id)
        return path_dis, path_list

    elif return_path is False:
        path_dis = nx.single_source_dijkstra_path_length(G, base_id)
        return path_dis


def add_nodes(G, base_node, indices, distance, threshold):

    """
    Adds a set of nodes and weighted edges based on pairs of indices
    between base_node and all entries in indices. Each node pair shares an
    edge with weight equal to the distance between both nodes.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object to which all nodes/edges will be added.
    base_node : int
        Base node's id to be added. All other nodes will be paired with
        base_node to form different edges.
    indices : list or array
        Set of nodes indices to be paired with base_node.
    distance : list or array
        Set of distances between all nodes in 'indices' and base_node.
    threshold : float
        Edge distance threshold. All edges with distance larger than
        'threshold' will not be added to G.

    """

    for c in np.arange(len(indices)):
        if distance[c] <= threshold:
            # If the distance between vertices is less than a given
            # threshold, add edge (i[0], i[c]) to Graph.
            G.add_weighted_edges_from([(base_node, indices[c],
                                        distance[c])])
