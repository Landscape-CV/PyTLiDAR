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
Date: 9 Feb 2025
Copyright (C) 2025 Georgia Institute of Technology Human-Augmented Analytics Group

This derivative work is released under the GNU General Public License (GPL).
"""

import numpy as np
from scipy.spatial.distance import cdist
from Utils.Utils import unique_elements_array
def segments(cover, Base, Forb,qsm=True):
    """
    Segments the covered point cloud into branches.

    Inputs:
    cover         : Cover sets
    Base          : Base of the tree
    Forb          : Cover sets not part of the tree

    Outputs:
    segment       : Dictionary containing the following fields:
        segments      : Segments found, list of lists, each list contains the cover sets
        ParentSegment : Parent segment of each segment, list of integers,
                        equals to zero if no parent segment
        ChildSegment  : Children segments of each segment, list of lists
    """

    Nei = cover['neighbor']
    nb = len(Nei)  # The number of cover sets
    a = max(200000, nb // 100)  # Estimate for maximum number of segments
    SBas = [None] * a  # The segment bases found
    Segs = [None] * a  # The segments found
    SPar = np.zeros((a, 2), dtype=int)-1 # The parent segment of each segment
    SChi = [None] * a  # The children segments of each segment

    # Initialize SChi
    SChi[0] = np.zeros(5000, dtype=np.uint32)
    C = np.zeros(1000, dtype=np.uint32)
    for i in range(1, a):
        SChi[i] = C.copy()
    NChi = np.zeros(a, dtype=np.uint32)  # Number of child segments found for each segment

    Fal = np.zeros(nb, dtype=bool)  # Logical false-vector for cover sets
    s = 0  # The index of the segment under expansion
    b = s  # The index of the latest found base

    SBas[s] = Base
    Seg = [None] * 1000  # The cover set layers in the current segment
    Seg[0] = Base

    ForbAll = Fal.copy()  # The forbidden sets
    ForbAll[Forb] = True
    ForbAll[Base] = True
    Forb = ForbAll.copy()  # The forbidden sets for the segment under expansion

    Continue = True  # True as long as the component can be segmented further
    NewSeg = True  # True if the first Cut for the current segment
    nl = 0  # The number of cover set layers currently in the segment

    # Segmenting stops when there are no more segments to be found
    ind = 0
    while Continue and (b < nb):
        ind+=1
        # Update the forbidden sets
        
        Forb[Seg[nl]] = True
        # print(ind,nl,sum(Forb))
        # Define the study
        Cut = define_cut(Nei, Seg[nl], Forb, Fal)
        CutSize = len(Cut)

        if NewSeg:
            NewSeg = False
            ns = min(CutSize, 6)

        # Define the components of cut and study regions
        if CutSize > 0:
            CutComps, _CompSize = cut_components(Nei, Cut, CutSize, Fal, Fal)
            nc = len(CutComps)
            if nc > 1:
                StudyComps, Bases, CompSize, Cont, BaseSize = study_components(
                    Nei, ns, Cut, CutComps, Forb, Fal, Fal
                )
                nc = len(Cont)
        else:
            nc = 0

        # Classify study region components
        if nc == 1:
            # One component, continue expansion of the current segment
            nl += 1
            if len(Cut.shape)>1 and Cut.shape[1] >1:
                Seg[nl] = Cut.T
            else:
                Seg[nl] = Cut
        elif nc > 1:
            # Classify the components of the Study region
            
            Class = component_classification(CompSize, Cont, BaseSize, CutSize,qsm)

            for i in range(nc):
                if Class[i] == 1:
                    Base = Bases[i].copy()
                    ForbAll[Base] = True
                    Forb[StudyComps[i]] = True
                    J = Forb[Cut]
                    Cut = Cut[~J].copy()
                    b += 1
                    SBas[b] = Base
                    SPar[b, :] = [s, nl]
                    NChi[s] += 1
                    SChi[s][NChi[s] - 1] = b

            # Define the new cut.
            # If the cut is empty, determine the new base
            if len(Cut) == 0:
                Segs[s] = np.array(Seg[:nl+1], dtype = 'object')
                S = np.concatenate(Seg[:nl+1])
                ForbAll[S] = True

                if s < b:
                    s += 1
                    Seg[0] = SBas[s].copy()
                    Forb = ForbAll.copy()
                    NewSeg = True
                    nl = 0
                else:
                    Continue = False
            else:
                if len(Cut.shape)>1 and Cut.shape[1] >1:
                    Cut = Cut.T
                nl += 1
                Seg[nl] = Cut
        else:
            # If the study region has zero size, then the current segment is
            # complete and determine the base of the next segment
            Segs[s] = np.array(Seg[:nl+1], dtype = 'object')
            S = np.concatenate(Seg[:nl+1])
            ForbAll[S] = True

            if s < b:
                s += 1
                Seg[0] = SBas[s]
                Forb = ForbAll.copy()
                NewSeg = True
                nl = 0
            else:
                Continue = False
    b+=1
    Segs = Segs[:b]
    
    SPar = SPar[:b, :]
    schi = SChi[:b]

    # Define output
    assigned_segs = np.zeros(np.max(cover["sets"])+1)-1
    SChi = [None] * b
    for i in range(b):
        if NChi[i] > 0:
            SChi[i] = schi[i][:NChi[i]].astype(np.uint32)
        else:
            SChi[i] = np.zeros(0, dtype=np.uint32)
        S = Segs[i]
        for j in range(len(S)):
            S[j] = S[j].astype(np.uint32)
            assigned_segs[S[j].astype(np.uint32)] = i

        Segs[i] = S

    segment = {
        'segments': Segs,
        'ParentSegment': SPar,
        'ChildSegment': SChi,
        'SegmentArray': assigned_segs
    }

    return segment

def define_cut(Nei,CutPre,Forb,Fal):
    """% Defines the "Cut" region

    Args:
        Nei (_type_): _description_
        CutPre (_type_): _description_
        Forb (_type_): _description_
        Fal (_type_): _description_
    """
    
    Cut = np.concatenate([Nei[c] for c in CutPre])
    Cut = unique_elements_array(Cut,Fal)
    I = Forb[Cut]
    Cut = Cut[np.invert(I)]
    return Cut

def cut_components(Nei, Cut, CutSize, Fal, False_mask):
    """
    Defines the connected components of the Cut region.

    Inputs:
    Nei       : Neighbor list for each cover set (list of lists or list of arrays)
    Cut       : Indices of the cover sets in the Cut region
    CutSize   : Number of cover sets in the Cut region
    Fal       : Boolean mask indicating forbidden cover sets
    False_mask: Boolean mask for additional filtering

    Outputs:
    Components: List of connected components (list of arrays)
    CompSize  : Number of cover sets in each component (list of integers)
    """
    Fal = Fal.copy()
    False_mask= False_mask.copy()
    Cut = Cut.copy()
    if CutSize == 1:
        # Cut is connected and therefore Study is also
        CompSize = 1
        Components = [Cut]
    elif CutSize == 2:
        I = Nei[Cut[0]] == Cut[1]
        if np.any(I):
            Components = [Cut]
            CompSize = 1
        else:
            Components = [Cut[0], Cut[1]]
            CompSize = [1, 1]
    elif CutSize == 3:
        I = Nei[Cut[0]] == Cut[1]
        J = Nei[Cut[0]] == Cut[2]
        K = Nei[Cut[1]] == Cut[2]
        if np.sum(I) + np.sum(J) + np.sum(K) >= 2:
            CompSize = 1
            Components = [Cut]
        elif np.any(I):
            Components = [Cut[:2], Cut[2]]
            CompSize = [2, 1]
        elif np.any(J):
            Components = [Cut[[0, 2]], Cut[1]]
            CompSize = [2, 1]
        elif np.any(K):
            Components = [Cut[1:3], Cut[0]]
            CompSize = [2, 1]
        else:
            CompSize = [1, 1, 1]
            Components = [Cut[0], Cut[1], Cut[2]]
    else:
        Components = [None] * CutSize
        CompSize = np.zeros(CutSize, dtype=int)
        Comp = np.zeros(CutSize, dtype=int)
        Fal[Cut] = True
        nc = 0  # number of components found
        m = Cut[0]
        i = 0
        while i  < CutSize:
            Added = Nei[m]
            I = Fal[Added]
            Added = Added[I]
            a = len(Added)
            Comp = np.zeros(CutSize, dtype=int)
            Comp[0] = m
            Fal[m] = False
            t = 1
            num_added = 1
            while a > 0 :
                if t+a > len(Comp):
                    Comp = np.concatenate([Comp,np.zeros((t+a-len(Comp)))])
                Comp[t : t + a] = Added
                Fal[Added] = False
                t += a
                Ext = np.concatenate([Nei[a] for a in Added])
                Ext = unique_elements_array(Ext, False_mask)
                I = Fal[Ext]
                Added = Ext[I]
                num_added += a
                a = len(Added)
                test_breakpoint = None

            i += t
            nc += 1
            Components[nc - 1] = Comp[:t]
            CompSize[nc - 1] = t
            if i < CutSize:
                J = Fal[Cut]
                m = Cut[J]
                m = m[0]

        Components = Components[:nc]
        CompSize = CompSize[:nc]

    return Components, CompSize


def study_components(Nei, ns, Cut, CutComps, Forb, Fal, False_mask):
    """
    Defines the study region and its components.

    Inputs:
    Nei       : Neighbor list for each cover set (list of lists or list of arrays)
    ns        : Number of layers in the study region
    Cut       : Indices of the cover sets in the Cut region
    CutComps  : Connected components of the Cut region (list of arrays)
    Forb      : Boolean mask indicating forbidden cover sets
    Fal       : Boolean mask for additional filtering
    False_mask: Boolean mask for additional filtering

    Outputs:
    Components: List of connected components in the study region (list of arrays)
    Bases     : List of base sets for each component (list of arrays)
    CompSize  : Number of cover sets in each component (list of integers)
    Cont      : Boolean array indicating if each component can be extended
    BaseSize  : Number of cover sets in the base of each component (list of integers)
    """

    # Define Study as a list of arrays
    Fal = Fal.copy()
    False_mask = False_mask.copy()
    Forb = Forb.copy()
    Study = [None] * ns
    StudySize = np.zeros(ns, dtype=int)
    Study[0] = Cut
    StudySize[0] = len(Cut)

    if ns >= 2:
        N = Cut
        i = 0
        while i < ns-1:
            Forb[N] = True
            N = np.concatenate([Nei[n] for n in N])
            N = unique_elements_array(N, Fal)
            I = Forb[N]
            N = N[~I]
            if len(N) > 0:
                i += 1
                Study[i] = N.copy()
                StudySize[i] = len(N)
            else:
                Study = Study[:i+1]
                StudySize = StudySize[:i+1]
                i = ns

    # Define study as a vector
    ns = len(StudySize)
    studysize = np.sum(StudySize)
    study = np.concatenate(Study)

    # Determine the components of study
    nc = len(CutComps)
    i = 0  # index of cut component
    j = 0  # number of elements attributed to components
    k = 0  # number of study components
    Fal[study] = True
    Components = [None] * nc
    CompSize = np.zeros(nc, dtype=int)
    Comp = np.zeros(studysize, dtype=int)

    while i < nc:
        C = CutComps[i]
        while j < studysize:
            a = len(C) if type(C) == np.ndarray else 1
            Comp = Comp.copy()
            Comp[:a] = C
            Fal[C] = False
            if a > 1:
                Add = unique_elements_array(np.concatenate([Nei[c] for c in C]), False_mask)
            else:

                Add = Nei[C]
                if len(Nei[C])>0 and type(Nei[C][0])==np.ndarray:
                    Add = Nei[C][0]
            t = a
            I = Fal[Add]
            Add = Add[I]
            a = len(Add)
            
            while a > 0 :#and t+a<len(Comp)

                if t+a > len(Comp):
                    Comp = np.concatenate([Comp,np.zeros((t+a-len(Comp)))])
                Comp[t : t + a] = Add
                Fal[Add] = False
                t += a
                Add = np.concatenate([Nei[a] for a in Add])
                Add = unique_elements_array(Add, False_mask)
                I = Fal[Add]
                Add = Add[I]
                a = len(Add)
            
            k += 1
            Components[k - 1] = Comp[:t]
            j += t
            CompSize[k - 1] = t
            if j < studysize:
                C = np.zeros(0, dtype=int)
                while i < nc-1  and len(C) == 0:
                    i += 1
                    C = CutComps[i].astype(int)
                    J = Fal[C]
                    C = C[J]
                if i == nc-1  and len(C) == 0:
                    j = studysize
                    i = nc
            else:
                i = nc
        Components = Components[:k]
        CompSize = CompSize[:k]

    # Determine BaseSize and Cont
    Cont = np.ones(k, dtype=bool)
    BaseSize = np.zeros(k, dtype=int)
    Bases = [None] * k

    if k > 1:
        Forb[study] = True
        Fal[study] = False
        Fal[Study[0]] = True
        for i in range(k):
            # Determine the size of the base of the components
            Set = unique_elements_array(np.concatenate([Components[i], Study[0]]), False_mask).astype(int)
            False_mask[Components[i].astype(int)] = True
            I = (False_mask[Set] & Fal[Set]).copy()
            False_mask[Components[i].astype(int)] = False
            Set = Set[I]
            Bases[i] = Set
            BaseSize[i] = len(Set)
        Fal[Study[0]] = False
        Fal[Study[ns - 1]] = True
        Forb[study] = True
        for i in range(k):
            # Determine if the component can be extended
            Set = unique_elements_array(np.concatenate([Components[i], Study[ns - 1]]), False_mask).astype(int)
            False_mask[Components[i].astype(int)] = True
            I = False_mask[Set] & Fal[Set]
            False_mask[Components[i].astype(int)] = False
            Set = Set[I]
            if len(Set) > 0:
                N = np.concatenate([Nei[s] for s in Set])
                N = unique_elements_array(N, False_mask)
                I = Forb[N]
                N = N[~I]
                if len(N) == 0:
                    Cont[i] = False
            else:
                Cont[i] = False

    return Components, Bases, CompSize, Cont, BaseSize

def component_classification(CompSize, Cont, BaseSize, CutSize, qsm =True,trunk = False):
    """
    Classifies study region components into "continuation" or "branch".

    Inputs:
    CompSize : Number of cover sets in each component (list or array of integers)
    Cont     : Boolean array indicating if each component can be extended
    BaseSize : Number of cover sets in the base of each component (list or array of integers)
    CutSize  : Number of cover sets in the Cut region (integer)

    Outputs:
    Class    : Classification of components (list or array of integers)
                - Class[i] == 0: Continuation
                - Class[i] == 1: Branch
    """

    nc = len(CompSize)  # Number of components
    StudySize = np.sum(CompSize)  # Total number of cover sets in the study region
    Class = np.ones(nc, dtype=int)  # Initialize all components as branches
    ContiComp = -1  # Index of the continuation component (if any)

    # Simple initial classification
    for i in range(nc):
        if BaseSize[i] == CompSize[i] and not Cont[i]:
            # Component has no expansion, not a branch
            Class[i] = 0
        elif BaseSize[i] == 1 and CompSize[i] <= 2 and not Cont[i]:
            # Component has very small expansion, not a branch
            Class[i] = 0
        elif BaseSize[i] / CutSize < 0.05 and 2 * BaseSize[i] >= CompSize[i] and not Cont[i]:
            # Component has very small expansion or is very small, not a branch
            Class[i] = 0
        elif CompSize[i] <= 3 and not Cont[i]:
            # Very small component, not a branch
            Class[i] = 0
        elif BaseSize[i] / CutSize >= 0.7 or CompSize[i] >= 0.7 * StudySize:
            # Continuation of the segment
            
            Class[i] = 0
            ContiComp = i
        else:
            # Component is probably a branch
            pass

    # If no continuation component is found and there are branches,
    # mark the largest branch as the continuation
    Branches = Class == 1
    if ContiComp == -1 and np.any(Branches) and (qsm or trunk):
        Ind = np.arange(nc)  # Indices of all components
        Branches = Ind[Branches]  # Indices of branch components
        I = np.argmax(CompSize[Branches])  # Index of the largest branch
        Class[Branches[I]] = 0  # Mark the largest branch as continuation

    return Class



    