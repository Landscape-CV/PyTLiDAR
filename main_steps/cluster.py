
import numpy as np
import scipy.stats as stats
from main_steps.cover_sets import cover_sets
from main_steps.tree_sets import tree_sets
# from main_steps.segments import segments
import torch
import open3d as o3d
import numba 
from igraph import Graph



def get_segments(tile,initial_clusters,inputs):
    """
    gets segments as clusters using a QSM (Quantitative Structure Model) from the given tile and initial clusters.

    Parameters:
    - tile: The tile object containing the cover information.
    - initial_clusters: The initial clusters to be used for creating the QSM.

    Returns:
    - None
    """
    # inputs = {'PatchDiam1': 0.05, 'BallRad1':.075, 'nmin1': 5}
    # Create cover sets
    distinct_objects = np.unique(initial_clusters)
    labels = np.zeros(len(initial_clusters), dtype=int)
    for obj in distinct_objects:
        I = initial_clusters[obj]
        

    tile.cover_sets = cover_sets(tile, initial_clusters)
    
    # Create tree sets
    tile.tree_sets = tree_sets(tile, initial_clusters)
    
    # Create segments
    # tile.segments = segments(tile, initial_clusters)
    
    # Connect covers to neighbors
    connect_covers(tile, tile.Neighbors, initial_clusters, tile.cover_sets)

def connect_covers(tile, cover,  z_thresh=2, max_dist =.25):
    """
    Connects the covers of a tile to its neighbors.

    Parameters:
    - tile: The tile object containing the cover information.
    - Nei: The neighbor object containing the neighbor information.

    Returns:
    - None
    """
    
    Nei = cover['neighbor']
    set_points = torch.tensor(cover['ball'].astype(int),device=tile.device)
    npoints = len(tile.point_data)
    nb = len(Nei)
    idx =0
    NotExa = torch.ones(len(Nei), dtype=bool,device=tile.device)
    C = np.array([])
    tile.cover_sets = torch.Tensor(tile.cover_sets).to(tile.device).to(int)
    # updated_set = tile.cover_sets.copy()
    center_points = {}
    set_masks = torch.arange(nb)
    sets = torch.unique(tile.cover_sets)
    # center_points = torch.stack([
    #     torch.mean(tile.point_data[tile.cover_sets==m], dim=0) for m in set_masks
    # ])

    I = torch.argsort(tile.cover_sets)
    tile.cover_sets = tile.cover_sets[I]
    tile.point_data = tile.point_data[I]
    tile.cloud = tile.cloud[I]


    # unique_masks, inverse_indices = torch.unique(tile.cover_sets, return_inverse=True)

    num_masks = len(set_masks)
    dim = tile.point_data.size(1)


    center_points = torch.zeros((num_masks, dim), device=tile.point_data.device)
    center_points.scatter_reduce_(
    0, 
    tile.cover_sets.unsqueeze(-1).expand(-1, dim), 
    tile.point_data, 
    reduce='mean',
    include_self=False
    )


    std = torch.std(center_points[:,3])
    # points_in_set = torch.ones(nb,device=tile.device)#torch.tensor([len(n) for n in Nei], device = tile.device)
    points_in_set = torch.bincount(set_points)

    new_set = torch.arange(nb, device = tile.device).to(int)

    # again = True
    while NotExa.any() and idx < len(NotExa):
        # if idx > len(NotExa)-1:
            
        #     idx = int(torch.min(torch.where(NotExa)[0]))
            

        if NotExa[idx] == False:
            idx += 1
            continue
        try:
            C = Nei[idx]
        except:
            print(len(Nei),idx,nb, len(NotExa))
        C = torch.tensor(Nei[idx].astype(int),device = tile.device)#.to(tile.device)

        center_point = center_points[idx]
        changed = torch.zeros(nb, dtype=bool, device=tile.device)
        points_in_set = torch.ones(nb,device=tile.device)#torch.tensor([len(n) for n in Nei], device = tile.device)

        if len(C)>0:
            C = new_set[C.to(int)]
            centers = center_points[C]
            new_C = np.array([],dtype = np.float32)
            # distances = torch.norm(centers[:,0:3] - center_point[0:3], dim=1)
            
            # close_enough = distances <= max_dist 
            # if sum(close_enough)==0:
            #     C = []
            #     NotExa[idx]=False
            #     idx+=1
            #     continue
            # Z = (centers[:,3] - center_point[3])/std
            # Z = torch.abs(Z)
            # same_threshold = Z <= z_thresh
            # close_enough = close_enough & same_threshold
            # if sum(close_enough)==0:
            #     C = []
            #     NotExa[idx]=False
            #     idx+=1
            #     continue
            # center_points[idx] = torch.sum(centers[close_enough], dim = 0)/torch.sum(points_in_set[C[close_enough]])

            
            
            close_enough =torch.ones(len(C), dtype=bool, device=tile.device)   
            continuation = torch.ones(len(C[close_enough]), dtype=bool, device=tile.device)
            # for i,set in enumerate(C[close_enough]):

            #     if set == idx:
            #         continuation[i] = False
            #         continue
            #     if points_in_set[set] > points_in_set[idx]*4:
            #         continuation[i] = False





            for set in C[close_enough][continuation]:
                # if set == idx:
                #     continue
                new_set[set] = idx
                n = Nei[int(set)]
                if len(n)>0:

                    n = n[n!=set]
                    new_C = np.append(new_C,n)
                    NotExa[int(set)] = False
                    Nei[int(set)] = []
                    points_in_set[idx]+=points_in_set[set]
                    points_in_set[set] = 0
                    center_points[set] = center_points[idx]
                changed[set] = True
            weighted_sum = torch.sum(centers * close_enough.reshape(-1, 1), dim=0)  # Sum centers where mask=True
            total_weight = torch.sum(points_in_set[C] * close_enough)  # Sum weights where mask=True
            center_points[idx] = weighted_sum / total_weight
            # Nei[idx] = new_C
            C = torch.tensor(new_C.astype(int),device = tile.device)
        if len(C)==0:

            NotExa[idx] = False
        
        if torch.any(changed):
            I = torch.where(torch.isin(new_set, torch.where(changed)[0]))[0] #nodes changed to the indices that were changed this loop
            new_set[I] = idx
        idx+=1
    
    
    I = torch.argsort(tile.cover_sets)
    tile.cover_sets = tile.cover_sets[I]
    tile.point_data = tile.point_data[I]
    tile.cloud = tile.cloud[I]
    num_indices = torch.bincount(tile.cover_sets.to(int))
    if len(num_indices) < len(new_set):
        num_indices = torch.cat([num_indices, torch.zeros(len(new_set)-len(num_indices), device=tile.device,dtype=int)])
    
    tile.cover_sets = torch.repeat_interleave(new_set, num_indices)
    cover['neighbor'] = Nei
    


    
    # a = 0
    # for i,set in enumerate(new_set):
        
    #     t = num_indices[i] if len(num_indices)>i else 0
    #     if t == 0:
    #         continue
    #     if set != tile.cover_sets[a]:
    #         tile.cover_sets[a:t+a] = set
        
    #     a = t+a
        

     
    tile.cluster_labels = tile.cover_sets




def segment_point_cloud(tile, max_dist = .2, base_height = 1, layer_size =.25):
    tile.to(tile.device)
    tile.cover_sets = torch.Tensor(tile.cover_sets).to(tile.device).to(int)
    I = torch.argsort(tile.cover_sets)
    tile.cover_sets = tile.cover_sets[I]
    tile.point_data = tile.point_data[I]
    tile.cloud = tile.cloud[I]


    # unique_masks, inverse_indices = torch.unique(tile.cover_sets, return_inverse=True)

    num_masks = torch.max(tile.cover_sets)#torch.bincount(tile.cover_sets)
    dim = tile.point_data.size(1)


    center_points = torch.zeros((num_masks, dim), device=tile.point_data.device)
    center_points.scatter_reduce_(
    0, 
    tile.cover_sets.unsqueeze(-1).expand(-1, dim), 
    tile.point_data, 
    reduce='mean',
    include_self=False
    )
    
    
    
    cloud = center_points[:,:3].cpu().numpy()
    min_Z = np.min(cloud[:,2])
    I = (cloud[:,2]-min_Z)<base_height
    cloud = cloud[I]
    included_cover_sets = np.where(I)

    
    # cloud = tile.get_cloud_as_array()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    # pcd = pcd.voxel_down_sample(voxel_size=.03)
   
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    segments = np.zeros(len(cloud))-1

    not_explored = np.ones(len(cloud),dtype = bool)
    segment_num = 0
    
    neighbors = {}
    base = 0
    

        
  
    # segments,segment_num,not_explored = add_layer(pcd_tree,pcd,segments,not_explored,segment_num,max_dist)


    full_segments = np.zeros(len(center_points))-1
    # full_segments[included_cover_sets] = segments
    # segments = full_segments.copy()
    # full_not_explored = np.ones(len(center_points),dtype = bool)
    # full_not_explored[included_cover_sets]= not_explored
    # not_explored = full_not_explored.copy()


    I = torch.argsort(center_points[:,2])
    center_points = center_points[I]
    I = I.cpu().numpy()
    # not_explored = not_explored[I]
    # segments=segments[I]
    sorted_indices = I.copy()

    cloud = center_points[:,:3].cpu().numpy()

    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(cloud)
    full_pcd_tree = o3d.geometry.KDTreeFlann(full_pcd)
    min_Z = np.min(cloud[:,2])
    I = (cloud[:,2]-min_Z)<base_height
    
    prev_base_height = 0
    network = Graph()
    network.add_vertices(len(center_points))
    print("Build Networks")
    size_limit = 1000
    multiplier = 2
    while base_height+min_Z-1<torch.max(tile.cloud[:,2]):
        
        I = ((cloud[:,2]-min_Z)<base_height) & (prev_base_height<(cloud[:,2]-min_Z))
        
        
        cloud = cloud[I]
        included_cover_sets = np.where(I)
        segments = np.zeros(len(cloud))-1

        not_explored = np.ones(len(cloud),dtype = bool)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        # pcd = pcd.voxel_down_sample(voxel_size=.03)
    
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        segments,segment_num,not_explored = add_layer(pcd_tree,pcd,segments,not_explored,segment_num,max_dist*multiplier,network,full_pcd_tree,full_pcd,included_cover_sets[0],size_limit)
        full_segments[included_cover_sets] = segments
        if prev_base_height ==0:
            tree_bases = np.unique(segments)
        # full_not_explored[included_cover_sets]= not_explored
        cloud = center_points[:,:3].cpu().numpy()
        prev_base_height = base_height
        base_height+=layer_size
        size_limit = 50
        multiplier =1
    segments = full_segments.copy()
    full_not_explored = np.ones(len(center_points),dtype = bool)
    cloud = center_points[:,:3].cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
        # pcd = pcd.voxel_down_sample(voxel_size=.03)
    
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    #
    filtered_tree_bases = []
    for base in tree_bases:
        base_set = center_points[segments ==base]
        if  len(base_set[:,2]<min_Z+.3)>25:

            filtered_tree_bases.append(base)
    

    
    
    print("Connect Segments")
    segments,not_explored = connect_segments(pcd_tree,pcd,segments,full_not_explored,filtered_tree_bases,max_dist*1.5,network,False)
    segments,not_explored = connect_segments(pcd_tree,pcd,segments,not_explored,filtered_tree_bases,max_dist*1.5,network,False)
    

    unassigned_sets = np.where(~np.isin(segments,filtered_tree_bases))
    segments[unassigned_sets]=-1


    I = torch.argsort(tile.cover_sets)
    tile.cover_sets = tile.cover_sets[I]
    tile.point_data = tile.point_data[I]
    tile.cloud = tile.cloud[I]
    num_indices = torch.bincount(tile.cover_sets.to(int))
    segments = segments[np.argsort(sorted_indices)]
    segments=torch.tensor(segments,device=tile.device,dtype=int)
    if len(num_indices) < len(segments):
        num_indices = torch.cat([num_indices, torch.zeros(len(segments)-len(num_indices), device=tile.device,dtype=int)])
    if len(segments)<len(num_indices):
        segments = torch.cat([segments, torch.zeros(len(num_indices)-len(segments), device=tile.device,dtype=int)])
    
    tile.segment_labels= torch.repeat_interleave(segments, num_indices)
    tile.cover_sets = tile.cover_sets.cpu().numpy()
    tile.numpy()
    # tile.cluster_labels = segments
    # tile.cloud = cloud
    

# @numba.jit(forceobj=True)
def add_layer(pcd_tree,pcd,segments,not_explored,segment_num,max_dist,network:Graph,full_pcd_tree,full_pcd,included_sets,size_limit = 100):
    K=20
    edges = []
    weights = []

    while any(not_explored):
        
        # print(f"segment: {segment_num}, remaining: {sum(not_explored)}")
        base = np.min(np.where(not_explored))
        not_explored[base]=False
        
        k,points,dist = pcd_tree.search_hybrid_vector_3d(pcd.points[base],max_dist,K)
        k,graph_points,dist =full_pcd_tree.search_hybrid_vector_3d(full_pcd.points[included_sets[base]],max_dist*3,K)
        edges.extend(make_edges(included_sets[base],list(graph_points)))
        
        weights.extend(list(dist))
        
        points.pop(0)
        segments[base] = segment_num
        p_arr = np.array(points)
        I = not_explored[p_arr]
        points = p_arr[I]
        points = o3d.utility.IntVector(points)
        point_count = len(points)
        while len(points)>0 and point_count<size_limit:
            
            next_point = points.pop()
            if not_explored[next_point]:
                segments[next_point]=segment_num
                k,new_points,dist = pcd_tree.search_hybrid_vector_3d(pcd.points[next_point],max_dist,K)
                k,graph_points,dist =full_pcd_tree.search_hybrid_vector_3d(full_pcd.points[included_sets[next_point]],max_dist*3,K)
                edges.extend(make_edges(included_sets[next_point],list(graph_points)))
                weights.extend(list(dist))
                # new_points.pop(0)
                not_explored[next_point]=False
                
                

                new_points = np.setdiff1d(new_points,points)
                I = not_explored[new_points]
                new_points = new_points[I]
                points.extend(new_points)
                point_count = point_count + len(new_points)

                
            
        segment_num= segment_num+1
    network.add_edges(edges,{"weight":weights})
    return segments,segment_num,not_explored

@numba.jit(nopython=True)
def make_edges(source,target):
    edges= []
    for node in target:
        edges.append((source,node))
    return edges
# @numba.jit(forceobj=True)
def connect_segments(pcd_tree,pcd,segments,not_explored,tree_bases,max_dist,network,search_non_connecting):
    not_expanded = np.zeros(len(not_explored),dtype=bool)

    point_data = np.array(pcd.points)
    tree_base_points = []
    for base in tree_bases:
        # lexord = (point_data[np.where(segments == base)][:,0],point_data[np.where(segments == base)][:,1],point_data[np.where(segments == base)][:,2])
        base_set = point_data[np.where(segments == base)]
        base_estimate = np.lexsort((base_set[:,0],base_set[:,1],base_set[:,2]))[len(base_set)//2]
        tree_base_points.append(np.where(segments == base)[0][base_estimate])
        # tree_base_points.append(np.where(segments == base)[0][point_data[np.where(segments == base)][:,2].argmin()])

    tree_base_points=np.array(tree_base_points,dtype=int)
    tree_bases=np.array(tree_bases,dtype = int)
    

    while any(not_explored):
        
        # print(f"segment: {segment_num}, remaining: {sum(not_explored)}")
        base = np.min(np.where(not_explored))
        not_explored[base]=False
        if segments[base] in tree_bases:
            continue
        k,points,_ = pcd_tree.search_radius_vector_3d(pcd.points[base],max_dist)
        
        
        segs = np.unique(segments[points])
        
        if len(segs)==1:
            not_expanded[base]=True
            continue
        

        else:
            
            base_seg = segments[base]
            tree_base_seg =np.intersect1d(segs,tree_bases).astype(int)
            if len(tree_base_seg)==1:
                base_seg = tree_base_seg[0]
            elif len(tree_base_seg)==0:
                if not search_non_connecting:
                    not_expanded[base]=True
                    continue
                else:
                    path_dist=np.array(network.distances(base,tree_base_points,weights='weight'))[0]
                    base_seg=np.argmin(path_dist)
            else:

                base_idx = np.where(np.isin(tree_bases, tree_base_seg))[0]

                # euc_dist = np.sqrt(np.array([(pcd.points[idx]- pcd.points[base])**2 for idx in base_idx]).sum(axis=1))
                # base_seg=tree_base_seg[np.argmin(euc_dist)]
                path_dist=np.array(network.distances(base,tree_base_points[base_idx],weights='weight'))[0]
                base_seg=tree_base_seg[np.argmin(path_dist)]
            # for seg in segs:
            #     if seg not in tree_bases:
            
            segments[segments==segments[base] ] = base_seg
        

                
            
        
        
    return segments,not_expanded







                
                

            
            

                        
                            
