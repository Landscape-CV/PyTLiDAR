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
from Utils.Utils import distances_to_line,mat_vec_subtraction ,set_difference
def correct_segments(P,cover,segment,inputs,RemSmall=None,ModBases=None,AddChild=None):
    if RemSmall is None:
        RemSmall = True
    if ModBases is None:
        ModBases = False
    if AddChild is None:
        AddChild = False
    Bal = cover['ball']
    Segs = segment['segments']
    SPar = segment['ParentSegment']
    SChi = segment['ChildSegment']
    Ce = P[cover['center'], :]

    # Make stem and branches as long as possible
    if RemSmall:
        Segs, SPar, SChi = modify_topology(P, Ce, Bal, Segs, SPar, SChi, inputs['PatchDiam2Max'])
    else:
        Segs, SPar, SChi = modify_topology(P, Ce, Bal, Segs, SPar, SChi, inputs['PatchDiam1'])

    # Remove small child segments
    if RemSmall:
        Segs, SPar, SChi = remove_small(Ce, Segs, SPar, SChi)

    # Check the consistency of empty vector sizes
    ns = len(Segs)
    for i in range(ns):
        if  len(SChi[i])==0:
            SChi[i] = np.zeros((0, 1), dtype=np.uint32)

    if ModBases:
        # Modify the base of the segments
        ns = len(Segs)
        base = [None] * 200
        if AddChild:
            # Add the expanded base to the child and remove it from the parent
            for i in range(1, ns):
                base = [None] * 200
                SegC = Segs[i].copy()
                SegP = Segs[int(SPar[i, 0])].copy()
                SegP, Base = modify_parent(P, Bal, Ce, SegP, SegC, SPar[i, 1], inputs['PatchDiam1'], base)
                Segs[int(SPar[i, 0])] = SegP.copy()
                SegC[0] = Base
                Segs[i] = SegC.copy()
        else:
            # Only remove the expanded base from the parent
            for i in range(1, ns):
                if SPar[i, 0] > 0:
                    SegC = Segs[i].copy()
                    SegP = Segs[SPar[i, 0]].copy()
                    SegP, Base = modify_parent(P, Bal, Ce, SegP, SegC, SPar[i, 1], inputs['PatchDiam2Max'], base)
                    Segs[SPar[i, 0]] = SegP.copy()
    SPar = SPar[:, 0]

    # Modify the size and type of SChi and Segs, if necessary
    ns = len(Segs)
    for i in range(ns):
        C = SChi[i]
        if len(C.shape)>1 and C.shape[1] > C.shape[0] and C.shape[0] > 0:
            SChi[i] = C.T.astype(np.uint32)
        elif C.shape[0] == 0 or (len(C.shape)>1 and C.shape[1] == 0):
            SChi[i] = np.zeros((0, 1), dtype=np.uint32)
        else:
            SChi[i] = C.astype(np.uint32)
        S = Segs[i]
        for j in range(len(S)):
            S[j] = S[j].astype(np.uint32)
        Segs[i] = S
    segment['segments'] = Segs
    segment['ParentSegment'] = SPar
    segment['ChildSegment'] = SChi

    # Generate segment data for the points
    nump = P.shape[0]
    ns = len(Segs)
    # Define for each point its segment
    Bal = np.array(Bal,dtype = 'object')
    if ns <= 2**16:
        SegmentOfPoint = np.zeros(nump, dtype=np.int16)-1
    else:
        SegmentOfPoint = np.zeros(nump, dtype=np.int32)-1
    for i in range(ns):
        S = Segs[i]
        S = np.concatenate([s for s in S])
        SegmentOfPoint[np.hstack(Bal[S])] = i
    segment['SegmentOfPoint'] = SegmentOfPoint
    # Define the indexes of the segments up to 3rd-order
    C = SChi[0]
    segment['branch1indexes'] = C
    if C.size > 0:
        segs = [SChi[c] for c in C if len(SChi[c])>0] 
        C = np.concatenate(segs) if len(segs)>0 else np.array([])
        segment['branch2indexes'] = C
        if C.size > 0:
            segs = [SChi[c] for c in C if len(SChi[c])>0]

            C = np.concatenate(segs) if len(segs)>0 else np.array([])
            segment['branch3indexes'] = C
        else:
            segment['branch3indexes'] = np.zeros((0, 1), dtype=np.uint32)
    else:
        segment['branch2indexes'] = np.zeros((0, 1), dtype=np.uint32)
        segment['branch3indexes'] = np.zeros((0, 1), dtype=np.uint32)

    return segment


def search_stem_top(P, Ce, Bal, Segs, SPar, dmin):
    """
    Search the stem's top segment such that the resulting stem:
    1) is one of the highest segments (goes to the top of the tree),
    2) is horizontally close to the bottom of the stem (goes straight up),
    3) has a length close to the distance between its bottom and top (is not too curved).

    Parameters:
    P (np.ndarray): Point cloud.
    Ce (np.ndarray): Centers of the cover sets.
    Bal (list): Ball (cover set) indices.
    Segs (list): Segments.
    SPar (np.ndarray): Parent segment information.
    dmin (float): Minimum distance parameter.

    Returns:
    StemTop (int): Index of the stem's top segment.
    """

    nseg = len(Segs)
    SegHeight = np.zeros(nseg)  # Heights of the tips of the segments
    HorDist = np.zeros(nseg)  # Horizontal distances of the tips from stem's center

    s = Segs[0][0]
    StemCen = np.mean(Ce[s, :], axis=0)  # Center (x, y) of stem base

    for i in range(nseg):
        S = Segs[i][-1][0]
        SegHeight[i] = Ce[S, 2]
        HorDist[i] = np.linalg.norm(Ce[S, :2] - StemCen[:2])

    Top = np.max(SegHeight)  # The height of the highest tip
    HeiDist = Top - SegHeight  # The height difference to "Top"
    Dist = np.sqrt(HorDist**2 + HeiDist**2)  # Distance to the top

    LenDisRatio = 2
    SearchDist = 0.5
    MaxLenDisRatio = 1.05  # The maximum acceptable length/distance ratio of segments
    SubSegs = np.zeros(100, dtype=int)-1  # Segments to be combined to form the stem

    while LenDisRatio > MaxLenDisRatio:
        SubSegs = SubSegs.copy()
        StemTops = np.arange(nseg)
        I = Dist < SearchDist  # Only segments with distance to the top < 0.5m

        while not np.any(I):
            SearchDist += 0.5
            I = Dist < SearchDist

        StemTops = StemTops[I]

        # Define i-1 alternative stems from StemTops
        n = len(StemTops)
        Stems = [None] * n
        

        for j in range(n):
            Seg = Segs[0].copy()
            spar = SPar.copy()
            Segment = [None] * 3000
            if StemTops[j] != 1:
                # Tip point was not in the current segment, modify segments
                SubSegs[0] = StemTops[j]
                nsegs = 0
                segment = StemTops[j]

                while segment != 0:
                    segment = SPar[segment, 0]
                    nsegs += 1
                    if len(SubSegs)<nsegs:
                        SubSegs = np.concatenate([SubSegs,np.zeros(nsegs-len(SubSegs))])
                    SubSegs[nsegs] = segment

                # Modify stem
                a = len(Seg)
                Segment[:a] = Seg.copy()
                a +=1

                for i in range(nsegs - 1):
                    I = SubSegs[nsegs - i -1]  # Segment to be combined to the first segment
                    J = SubSegs[nsegs - i - 2]  # Above segment's child to be combined next
                    SP = spar[I , 1]  # Layer index of the child in the parent
                    SegC = Segs[I ]
                    sp = spar[J , 1]  # Layer index of the child's child in the child

                    if SP >= a - 1:  # Use the whole parent
                        Segment[a:a + sp] = SegC[:sp]
                        spar[J , 1] = a + sp 
                        a = a + sp + 1
                    else:  # Use only bottom part of the parent
                        Segment[SP + 1:SP + sp + 1] = SegC[:sp]
                        a = SP + sp + 2
                        spar[J , 1] = SP + sp 

                    SubSegs[nsegs - i -1] = 0

                # Combine the last segment to the branch
                I = SubSegs[0]
                SP = spar[I , 1]
                SegC = Segs[I ]
                nc = len(SegC)

                if SP >= a - 1:  # Use the whole parent
                    Segment[a:a + nc] = SegC
                    a = a + nc 
                else:  # Divide the parent segment into two parts
                    Segment[SP + 1:SP + nc + 1] = SegC
                    a = SP + nc +1




                Stems[j] = Segment[:a]
                # Stems[j] = np.array([seg for seg in Segment if seg is not None],dtype = 'object')#Segment[:a]
            else:
                Stems[j] = Seg

        # Calculate the lengths of the candidate stems
        N = int(np.ceil(0.5 / dmin / 1.4))  # Number of layers used for linear length approximation
        Lengths = np.zeros(n)
        Heights = np.zeros(n)

        for i in range(n):
            Seg = Stems[i]
            ns = len(Seg)

            if np.ceil(ns / N) > np.floor(ns / N):
                m = int(np.ceil(ns / N))
            else:
                m = int(np.ceil(ns / N)) + 1

            Nodes = np.zeros((m, 3))

            for j in range(m):
                I = (j) * N
                if I >= ns:
                    I = ns - 1
                S = Seg[I]
                if  len(S) > 1:
                    Nodes[j, :] = np.mean(Ce[S.astype(int), :], axis=0)
                else:
                    S = Bal[S[0]]
                    Nodes[j, :] = np.mean(P[S, :], axis=0)

            V = Nodes[1:, :] - Nodes[:-1, :]
            Lengths[i] = np.sum(np.sqrt(np.sum(V**2, axis=1)))
            V = Nodes[-1, :] - Nodes[0, :]
            Heights[i] = np.linalg.norm(V)

        LenDisRatio = Lengths / Heights
        LenDisRatio_min_idx = np.argmin(LenDisRatio)
        LenDisRatio = LenDisRatio[LenDisRatio_min_idx]
        StemTop = StemTops[LenDisRatio_min_idx]
        SearchDist += 1

        if SearchDist > 3:
            MaxLenDisRatio = 1.1
            if SearchDist > 5:
                MaxLenDisRatio = 1.15
                if SearchDist > 7:
                    MaxLenDisRatio = 5

    return StemTop


def search_branch_top(P, Ce, Bal, Segs, SPar, SChi, dmin, BI):
    """
    Search the end segment for a branch such that the resulting branch:
    1) has a length close to the distance between its bottom and top,
    2) has a distance close to the farthest segment end.

    Parameters:
    P (np.ndarray): Point cloud.
    Ce (np.ndarray): Centers of the cover sets.
    Bal (list): Ball (cover set) indices.
    Segs (list): Segments.
    SPar (np.ndarray): Parent segment information.
    SChi (list): Child segment information.
    dmin (float): Minimum distance parameter.
    BI (int): Branch (segment) index.

    Returns:
    BranchTop (int): The index of the segment forming the tip of the branch.
    """

    # Define all the sub-segments of the given segment
    ns = len(Segs)
    Segments = np.zeros(ns, dtype=int)  # The given segment and its sub-segments
    Segments[0] = BI
    t = 1
    C = SChi[BI]

    while len(C) > 0:
        n = len(C)
        if t+n>len(Segments):
            Segments = np.concatenate([Segments,np.zeros(t+n-len(Segments))])
        Segments[t:t + n] = C
        C = np.hstack([SChi[c] for c in C])
        t += n

    if t > 1:
        t -= n
    t+=1
    Segments = Segments[:t]

    # Determine linear distances from the segment tips to the base of the given segment
    LinearDist = np.zeros(t)  # Linear distances from the base
    Seg = Segs[Segments[0] ]
    BranchBase = np.mean(Ce[Seg[0].astype(int), :], axis=0)  # Center of branch's base

    for i in range(t):
        Seg = Segs[Segments[i]]
        C = np.mean(Ce[Seg[-1].astype(int), :], axis=0)  # Tip
        LinearDist[i] = np.linalg.norm(C - BranchBase)

    # Sort the segments according to their linear distance, from longest to shortest
    sorted_indices = np.argsort(LinearDist)[::-1]
    LinearDist = LinearDist[sorted_indices]
    Segments = Segments[sorted_indices]

    # Define alternative branches from Segments
    Branches = [None] * t  # The alternative segments as cell layers
    SubSegs = np.zeros(100, dtype=int)-1  # Segments to be combined
    Segment = [None] * 3000

    for j in range(t):
        Seg = Segs[BI]
        spar = SPar.copy()

        if Segments[j] != BI:
            # Tip point was not in the current segment, modify segments
            SubSegs[0] = Segments[j].copy()
            k = 0
            S = int(Segments[j])

            while S != BI:

                S = int(SPar[S , 0])
                k += 1
                if len(SubSegs)<k+1:
                    SubSegs = np.concatenate([SubSegs,np.zeros(100)])
                SubSegs[k] = S

            # Modify branch
            a = len(Seg)
            Segment[:a] = Seg.copy()
            a = a

            for i in range(k - 1):
                I = SubSegs[k - i - 1]  # Segment to be combined to the first segment
                J = SubSegs[k - i - 2]  # Above segment's child to be combined next
                SP = int(spar[I , 1])  # Layer index of the child in the parent
                SegC = Segs[I ]
                sp = int(spar[J, 1])  # Layer index of the child's child in the child

                if SP >= a - 2:  # Use the whole parent
                    Segment[a:a + sp+1] = SegC[:sp+1].copy()
                    spar[J, 1] = a + sp 
                    a = a + sp +1
                else:  # Use only bottom part of the parent 
                    Segment[SP + 1:SP + sp + 1] = SegC[:sp+1].copy()
                    a = SP + sp + 2
                    spar[J, 1] = SP + sp+1

                SubSegs[k - i - 1] = 0

            # Combine the last segment to the branch
            I = SubSegs[0]
            SP = int(spar[I, 1])
            SegC = Segs[I]
            L = len(SegC)

            if SP >= a - 2:  # Use the whole parent
                Segment[a:a + L-1] = SegC.copy()
                a = a + L-1
                Branches[j] = Segment[:a+1]
            else:  # Divide the parent segment into two parts
                Segment[SP+1 :SP + L +1] = SegC.copy()
                a = SP + L

                Branches[j] = Segment[:a+1].copy()
        else:
            Branches[j] = Seg.copy()

    # Calculate the lengths of the candidate branches
    N = int(np.ceil(0.25 / dmin / 1.4))  # Number of layers used for linear length approximation
    i = 0  # Running index for while loop
    Continue = True  # Continue while loop as long as "Continue" is True
    Lengths = np.zeros(t)  # Linear lengths of the branches

    while i < t and Continue:
        # Approximate the length with line segments connecting nodes along the segment
        Seg = Branches[i]
        ns = len(Seg)

        if np.ceil(ns / N) > np.floor(ns / N):
            m = int(np.ceil(ns / N))
        else:
            m = int(np.ceil(ns / N)) + 1

        Nodes = np.zeros((m, 3))

        for j in range(m):
            I = j * N
            if I >= ns:
                I = ns - 1
            S = Seg[I]
            if len(S) > 1:
                Nodes[j, :] = np.mean(Ce[S.astype(int), :], axis=0)
            else:
                S = Bal[S[0]]
                Nodes[j, :] = np.mean(P[S, :], axis=0)

        V = Nodes[1:, :] - Nodes[:-1, :]  # Line segments
        Lengths[i] = np.sum(np.sqrt(np.sum(V**2, axis=1)))

        # Continue as long as the length is less than 20% longer than the linear distance
        # and the linear distance is over 75% of the maximum
        if Lengths[i] / LinearDist[i] < 1.20 and LinearDist[i] > 0.75 * LinearDist[0]:
            Continue = False
            BranchTop = Segments[i]

        i += 1

    # If no suitable segment was found, try first with less strict conditions,
    # and if that does not work, then select the one with the largest linear distance
    if Continue:
        L = Lengths / LinearDist
        i = 0

        while i < t and L[i] > 1.4 and LinearDist[i] > 0.75 * LinearDist[0]:
            
            i += 1

        if i < t:
            BranchTop = Segments[i]
        else:
            BranchTop = Segments[0]

    return BranchTop

def modify_topology(P, Ce, Bal, Segs, SPar, SChi, dmin):
    """
    Make stem and branches as long as possible by modifying the topology.

    Parameters:
    P (np.ndarray): Point cloud.
    Ce (np.ndarray): Centers of the cover sets.
    Bal (list): Ball (cover set) indices.
    Segs (list): Segments.
    SPar (np.ndarray): Parent segment information.
    SChi (list): Child segment information.
    dmin (float): Minimum distance parameter.

    Returns:
    Segs (list): Modified segments.
    SPar (np.ndarray): Modified parent segment information.
    SChi (list): Modified child segment information.
    """

    ns = len(Segs)
    Fal = np.zeros(2 * ns, dtype=bool)
    nc = int(np.ceil(ns / 5))
    SubSegments = np.zeros(nc, dtype=int)-1  # For searching sub-segments
    SegInd = 0  # The segment under modification
    UnMod = np.ones(ns, dtype=bool)
    UnMod[SegInd] = False
    BranchOrder = 0
    ChildSegInd = 0  # Index of the child segments under modification
    ind = 0
    while np.any(UnMod):
        # seg_lens = [len(seg) for seg in Segs]
        ind+=1
        ChildSegs = SChi[SegInd]  # Child segments of the segment under modification
        if len(ChildSegs.shape)>1 and ChildSegs.shape[0] < ChildSegs.shape[1]:
            ChildSegs = ChildSegs.T
            SChi[SegInd] = ChildSegs

        if len(Segs[SegInd])>0 and len(ChildSegs) > 0:
            if SegInd > 0 and BranchOrder > 1:  # 2nd-order and higher branches
                # Search the tip of the sub-branches with the biggest linear distance from the current branch's base
                SubSegments[0] = SegInd
                NSubSegs = 1
                while ChildSegs.size > 0:
                    n = ChildSegs.size
                    if len(SubSegments)<NSubSegs+n:
                        SubSegments = np.concatenate([SubSegments,np.zeros(NSubSegs+n-len(SubSegments))])
                    SubSegments[NSubSegs:NSubSegs + n] = ChildSegs
                    ChildSegs = np.hstack([SChi[c] for c in ChildSegs])
                    NSubSegs += n

                if NSubSegs > 1:
                    NSubSegs -= n

                # Find tip-points
                Top = np.zeros((NSubSegs+1, 3))
                for i in range(NSubSegs+1):
                    Top[i, :] = Ce[Segs[SubSegments[i].astype(int)][-1][0], :]

                # Define bottom of the branch
                BotLayer = Segs[SegInd][0].astype(int)
                Bottom = np.mean(Ce[BotLayer, :], axis=0)

                # End segment is the segment whose tip has the greatest distance to the bottom of the branch
                V = Top - Bottom
                d = np.sum(V**2, axis=1)
                I = np.argmax(d)
                TipSeg = int(SubSegments[I])

            elif SegInd > 0 and BranchOrder <= 1:  # First-order branches

                TipSeg = int(search_branch_top(P, Ce, Bal, Segs, SPar, SChi, dmin, SegInd))

            else:  # Stem
                TipSeg = int(search_stem_top(P, Ce, Bal, Segs, SPar, dmin))

            if TipSeg != SegInd:
                # Tip point was not in the current segment, modify segments
                SubSegments[0] = TipSeg
                NSubSegs = 1
                while TipSeg != SegInd:
                    TipSeg = int(SPar[TipSeg, 0])
                    NSubSegs += 1
                    if len(SubSegments)<NSubSegs:
                        SubSegments = np.concatenate([SubSegments,np.zeros(NSubSegs-len(SubSegments))])
                    SubSegments[NSubSegs - 1] = TipSeg

                # Refine branch
                for i in range(NSubSegs - 2):
                    I = int(SubSegments[NSubSegs - i - 2 ])  # Segment to be combined to the first segment
                    J = int(SubSegments[NSubSegs - i - 3])  # Above segment's child to be combined next
                    SP = int(SPar[I, 1])  # Layer index of the child in the parent
                    SegP = Segs[SegInd]
                    SegC = Segs[I].copy()
                    N = len(SegP)
                    sp = int(SPar[J, 1])  # Layer index of the child's child in the child

                    if SP >= N - 3:  # Use the whole parent
                        Segs[SegInd] = np.array([seg for seg in SegP]+[seg for seg in SegC[:sp+1]],dtype = 'object')#np.concatenate([SegP, np.concatenate([seg for seg in SegC[:sp+1]])])
                        if sp < len(SegC):  # Use only part of the child segment
                            Segs[I] = SegC[sp+1:]
                            SPar[I, 1] = N + sp

                            ChildSegs = SChi[I].copy()
                            K = SPar[ChildSegs, 1] <= sp
                            c = ChildSegs[~K].copy()
                            SChi[I] = c
                            SPar[c, 1] = SPar[c, 1] - sp -1
                            ChildSegs = ChildSegs[K].copy()
                            SChi[SegInd] = np.hstack([SChi[SegInd], ChildSegs])
                            SPar[ChildSegs, 0] = SegInd
                            SPar[ChildSegs, 1] = N + SPar[ChildSegs, 1] 

                        else:  # Use the whole child segment
                            Segs[I] = []
                            SPar[I, 0] = 0
                            UnMod[I] = False

                            ChildSegs = SChi[I].copy()
                            SChi[I] = np.zeros(0, dtype=int)
                            c = np.set_difference(SChi[SegInd], I,Fal)
                            SChi[SegInd] = np.hstack([c, ChildSegs])
                            SPar[ChildSegs, 0] = SegInd
                            SPar[ChildSegs, 1] = N + SPar[ChildSegs, 1] 

                        SubSegments[NSubSegs - i -2] = SegInd

                    else:  # Divide the parent segment into two parts
                        ns += 1
                        Segs.append(SegP[SP + 1:])  # The top part of the parent forms a new segment
                        SPar = np.concatenate([SPar, np.zeros(shape = (1,SPar.shape[1]))])
                        SPar[ns - 1, :] = [SegInd, SP]
                        UnMod = np.concatenate([UnMod,np.zeros(1,dtype=bool)])
                        UnMod[ns - 1] = True

                        Segs[SegInd] = np.array([seg for seg in SegP[:SP+1]] + [seg for seg in SegC[:sp+1]], dtype = 'object')


                        ChildSegs = SChi[SegInd].copy()
                        if len(ChildSegs.shape)>1 and ChildSegs.shape[0] < ChildSegs.shape[1]:
                            ChildSegs = ChildSegs.T
                        K = SPar[ChildSegs, 1] > SP
                        SChi[SegInd] = ChildSegs[~K].copy()
                        ChildSegs = ChildSegs[K].copy()
                        SChi.append( ChildSegs.copy())
                        SPar[ChildSegs, 0] = ns - 1
                        SPar[ChildSegs, 1] = SPar[ChildSegs, 1] - SP-1
                        SChi[SegInd] = np.hstack([SChi[SegInd], ns - 1]).copy()

                        if sp < len(SegC)-1:  # Use only part of the child segment
                            Segs[I] = SegC[sp+1:]
                            SPar[I, 1] = SP + sp +1

                            ChildSegs = SChi[I]
                            K = SPar[ChildSegs, 1] <= sp
                            SChi[I] = ChildSegs[~K].copy()
                            SPar[ChildSegs[~K], 1] = SPar[ChildSegs[~K], 1] - sp -1
                            ChildSegs = ChildSegs[K].copy()
                            SChi[SegInd] = np.hstack([SChi[SegInd], ChildSegs])
                            SPar[ChildSegs, 0] = SegInd
                            SPar[ChildSegs, 1] = SP + SPar[ChildSegs, 1]+1

                        else:  # Use the whole child segment
                            Segs[I] = []
                            SPar[I, 0] = 0
                            UnMod[I] = False

                            ChildSegs = SChi[I].copy()
                            c = set_difference(SChi[SegInd], I,Fal)
                            SChi[SegInd] = np.hstack([c, ChildSegs])
                            SPar[ChildSegs, 0] = SegInd
                            SPar[ChildSegs, 1] = SP + SPar[ChildSegs, 1]+1

                        SubSegments[NSubSegs - i - 2] = SegInd

                # Combine the last segment to the branch
                I = int(SubSegments[0])
                SP = int(SPar[I, 1])
                SegP = Segs[SegInd]
                SegC = Segs[I]
                N = len(SegP)

                if SP >= N - 4:  # Use the whole parent
                    Segs[SegInd] = np.array([seg for seg in SegP]+[seg for seg in SegC],dtype='object')
                    Segs[I] = []
                    SPar[I, 0] = 0
                    UnMod[I] = False

                    ChildSegs = SChi[I]
                    if len(ChildSegs.shape)>1 and ChildSegs.shape[0] < ChildSegs.shape[1]:
                        ChildSegs = ChildSegs.T
                    c = set_difference(SChi[SegInd], I,Fal)
                    SChi[SegInd] = np.hstack([c, ChildSegs])
                    SPar[ChildSegs, 0] = SegInd
                    SPar[ChildSegs, 1] = N + SPar[ChildSegs, 1] 

                else:  # Divide the parent segment into two parts
                    ns += 1
                    SP = int(SP)
                    Segs.append(SegP[SP + 1:])
                    
                    SPar = np.concatenate([SPar, np.zeros(shape = (1,SPar.shape[1]))])
                    SPar[ns - 1, :] = [SegInd, SP]

                    # Segs[SegInd] = np.array([SegP[:SP+1],np.concatenate([seg for seg in SegC])],dtype = 'object')#np.concatenate([seg for seg in SegP[:SP+1]]+ [seg for seg in SegC])
                    Segs[SegInd] = np.array([seg for seg in SegP[:SP+1]]+[seg for seg in SegC],dtype = 'object')
                    Segs[I] = []
                    SPar[I, 0] = 0
                    UnMod = np.concatenate([UnMod,np.zeros(1)]).astype(bool)
                    UnMod[ns - 1] = True
                    UnMod[I] = False

                    ChildSegs = SChi[SegInd]
                    K = SPar[ChildSegs, 1] > SP
                    SChi[SegInd] = np.hstack([ChildSegs[~K], ns - 1])
                    ChildSegs = ChildSegs[K]
                    SChi.append(ChildSegs)
                    SPar[ChildSegs, 0] = ns - 1
                    SPar[ChildSegs, 1] = SPar[ChildSegs, 1] - SP -1

                    ChildSegs = SChi[I]
                    c = set_difference(SChi[SegInd], I,Fal)
                    SChi[SegInd] = np.hstack([c, ChildSegs])
                    SPar[ChildSegs, 0] = SegInd
                    SPar[ChildSegs, 1] = SP + SPar[ChildSegs, 1] +1

            UnMod[SegInd] = False
        else:
            UnMod[SegInd] = False

        # Select the next branch, use increasing branching order
        if  BranchOrder > 0  and np.any(UnMod[SegChildren]):
            ChildSegInd += 1
            SegInd = SegChildren[ChildSegInd]
        elif BranchOrder == 0:
            BranchOrder += 1
            SegChildren = SChi[0]
            if SegChildren.size > 0:
                SegInd = SegChildren[0]
            else:
                UnMod[:] = False
        else:
            BranchOrder += 1
            i = 1
            SegChildren = SChi[0]
            while i < BranchOrder and SegChildren.size > 0:
                i += 1
                L = np.array([len(SChi[c]) for c in SegChildren])
                Keep = L > 0
                SegChildren = SegChildren[Keep]
                sc = [SChi[c] for c in SegChildren]
                SegChildren = np.hstack(sc) if len(sc)>0 else np.array([])
            I = UnMod[SegChildren].astype(bool) if len(SegChildren)>0 else np.array([False])
            if np.any(I):
                SegChildren = SegChildren[I]
                SegInd = SegChildren[0]
                ChildSegInd = 0

    # Modify indexes by removing empty segments
    Empty = np.array([len(s) > 0 for s in Segs])
    Segs = [s for s, e in zip(Segs, Empty) if e]
    Ind = np.arange(ns)
    n = np.sum(Empty)
    I = np.arange(n)
    Ind[Empty] = I
    SPar = SPar[Empty, :]
    J = SPar[:, 0] > 0
    SPar[J, 0] = Ind[SPar[J, 0].astype(int)]
    for i in range(ns):
        if Empty[i]:
            ChildSegs = SChi[i]
            if ChildSegs.size > 0:
                ChildSegs = Ind[ChildSegs]
                SChi[i] = ChildSegs
    SChi = [SChi[i] for i in range(ns) if Empty[i]]
    ns = n

    # Modify SChi
    for i in range(ns):
        ChildSegs = SChi[i].copy()
        if len(ChildSegs.shape)>1 and ChildSegs.shape[0] < ChildSegs.shape[1] and ChildSegs.shape[0] > 0:
            SChi[i] = ChildSegs.T
        elif  ChildSegs.shape[0] == 0 or (len(ChildSegs.shape)>1 and ChildSegs.shape[1] == 0):
            SChi[i] = np.zeros(0, dtype=int)-1
        Seg = Segs[i].copy()
        n = len(Seg)
        newSeg = np.zeros(shape = (len(Seg),),dtype = 'object')
        for j in range(n):
            # Seg[j] = Seg[j].astype('int') if type(Seg[j])==np.ndarray else np.array([Seg[j]]).astype('int')
            ChildSegs = Seg[j].copy() 
            
            if len(ChildSegs.shape)>1 and ChildSegs.shape[0] < ChildSegs.shape[1] and ChildSegs.shape[0] > 0:
                newSeg[j] = ChildSegs.T
            elif ChildSegs.shape[0] == 0 or (len(ChildSegs.shape)>1 and ChildSegs.shape[1] == 0):
                newSeg[j] = np.array([])#np.zeros(0, dtype='object')
            else:
                newSeg[j] = ChildSegs

                    
        
        Segs[i] = newSeg

    return Segs, SPar, SChi


def remove_small(Ce, Segs, SPar, SChi):
    """
    Removes small child segments.

    Parameters:
    Ce (np.ndarray): Centers of the cover sets.
    Segs (list): Segments.
    SPar (np.ndarray): Parent segment information.
    SChi (list): Child segment information.

    Returns:
    Segs (list): Modified segments.
    SPar (np.ndarray): Modified parent segment information.
    SChi (list): Modified child segment information.
    """


# computes and estimate for stem radius at the base
    Segment = Segs[0]  # current or parent segment
    ns = len(Segment)  # layers in the parent
    if ns > 10:
        EndL = 10  # ending layer index in parent
    else:
        EndL = ns
    End = np.mean(Ce[Segment[EndL-1].astype(int)], axis=0)  # Center of end layer
    Start = np.mean(Ce[Segment[0]], axis=0)  # Center of starting layer
    V = End - Start  # Vector between starting and ending centers
    V = V / np.linalg.norm(V)  # normalize
    Sets = np.concatenate(Segment[:EndL])
    d, _V,_H,_B = distances_to_line(Ce[Sets.astype(int)], V, Start)
    MaxRad = np.max(d)

    Nseg = len(Segs)
    Fal = np.zeros(Nseg, dtype=bool)
    Keep = np.ones(Nseg, dtype=bool)
    Sets = np.zeros(2000, dtype=int)
    SPar = SPar.astype(int)
    SChi = np.array(SChi,dtype = 'object')
    Segs = np.array(Segs,dtype = 'object')
    for i in range(Nseg):
        if Keep[i]:
            ChildSegs = SChi[i]  # child segments
            if len(ChildSegs)>0:  # child segments exists
                n = len(ChildSegs)  # number of children
                Segment = Segs[i]  # current or parent segment
                ns = len(Segment)  # layers in the parent
                for j in range(n):  # check each child separately
                    nl = SPar[ChildSegs[j], 1]  # the index of the layer in the parent the child begins
                    if nl > 10:
                        StartL = nl - 10  # starting layer index in parent
                    else:
                        StartL = 0
                    if ns - nl > 10:
                        EndL = nl + 10  # end layer index in parent
                    else:
                        EndL = ns-1
                    End = np.mean(Ce[Segment[int(EndL)].astype(int)], axis=0)
                    Start = np.mean(Ce[Segment[int(StartL)].astype(int)], axis=0)
                    V = End - Start  # Vector between starting and ending centers
                    V = V / np.linalg.norm(V)  # normalize

                    # cover sets of the child
                    ChildSets = Segs[ChildSegs[j]]
                    NL = len(ChildSets)
                    a = 0
                    for k in range(NL):
                        S = ChildSets[k]
                        if a+len(S) > len(Sets):
                            Sets = np.concatenate([Sets,np.zeros((len(S)+a-len(Sets)))])

                        Sets[a:a + len(S)] = S
                        a += len(S)
                    ChildSets = Sets[:a]

                    # maximum distance in child
                    d1, _V,_H,_B = distances_to_line(Ce[ChildSets.astype(int)], V, Start)
                    distChild = np.max(d1)

                    if distChild < MaxRad + 0.06:

                        # Select the cover sets of the parent between centers
                        NL = EndL - StartL + 1
                        a = 0
                        for k in range(NL):
                            S = Segment[StartL + k]
                            if a+len(S) > len(Sets):
                                Sets = np.concatenate([Sets,np.zeros((len(S)+a-len(Sets)))])
                            Sets[a:a + len(S)] = S
                            a += len(S)
                        ParentSets = Sets[:a]

                        # maximum distance in parent
                        distPar = np.max(distances_to_line(Ce[ParentSets.astype(int)], V, Start)[0])
                        if (distChild - distPar < 0.02) or (distChild / distPar < 1.2 and distChild - distPar < 0.06):
                            ChildChildSegs = SChi[ChildSegs[j]]
                            nc = len(ChildChildSegs)
                            if nc == 0:
                                # Remove, no child segments
                                Keep[ChildSegs[j]] = False
                                Segs[ChildSegs[j]] = np.zeros((0, 1))
                                SPar[ChildSegs[j], :] = np.zeros((1, 2))
                                SChi[i] = set_difference(ChildSegs, ChildSegs[j], Fal)
                            else:
                                L = np.concatenate(SChi[ChildChildSegs.astype(int)])  # child child segments
                                if not L.any():
                                    J = np.zeros(nc, dtype=bool)
                                    for k in range(nc):
                                        segment = Segs[ChildChildSegs[k]]
                                        if len(segment)<1:
                                            J[k] = True
                                        else:
                                            segment1 = np.append(np.concatenate(segment), ParentSets).astype(int)
                                            distSeg = np.max(distances_to_line(Ce[segment1], V, Start)[0])
                                            if (distSeg - distPar < 0.02) or (distSeg / distPar < 1.2 and distSeg - distPar < 0.06):
                                                J[k] = True
                                    if np.all(J):
                                        # Remove
                                        ChildChildSegs1 = np.concatenate([ChildChildSegs, [ChildSegs[j]]]).astype(int)
                                        nc = len(ChildChildSegs1)
                                        Segs[ChildChildSegs1] = [np.zeros((0, 1))] * nc
                                        Keep[ChildChildSegs1] = False
                                        SPar[ChildChildSegs1, :] = np.zeros((nc, 2))-1
                                        d = set_difference(ChildSegs, ChildSegs[j], Fal)
                                        SChi[i] = d
                                        SChi[ChildChildSegs1] = [np.zeros((0, 1))] * nc
            if i == 0:
                MaxRad = MaxRad / 2
    # Modify segments and their indexing
    Segs = [Segs[i] for i in range(Nseg) if Keep[i]]
    n = np.sum(Keep)
    Ind = np.arange( Nseg )
    J = np.arange( n )
    Ind[Keep] = J
    Ind[~Keep] = -1
    SPar = SPar[Keep, :]
    J = SPar[:, 0] > -1
    SPar[J, 0] = Ind[SPar[J, 0]]

    # Modify SChi
    for i in range(Nseg):
        if Keep[i]:
            ChildSegs = SChi[i]
            if len(ChildSegs)>0:
                ChildSegs = np.array([c for c in Ind[ChildSegs] if c > -1],dtype = int)
                if len(ChildSegs.shape)>1 and ChildSegs.shape[0] < ChildSegs.shape[1]:
                    SChi[i] = ChildSegs.T
                else:
                    SChi[i] = ChildSegs
            else:
                SChi[i] = np.zeros((0, 1))
    SChi = SChi[Keep]
    
    return Segs, SPar, SChi


def modify_parent(P, Bal, Ce, SegP, SegC, nl, PatchDiam, base):
    """
    Expands the base of the branch backwards into its parent segment and
    then removes the expansion from the parent segment.

    Parameters:
    P (np.ndarray): Point cloud.
    Bal (list): Ball (cover set) indices.
    Ce (np.ndarray): Centers of the cover sets.
    SegP (list): Parent segment.
    SegC (list): Child segment.
    nl (int): Layer index in the parent segment.
    PatchDiam (float): Patch diameter parameter.
    base (list): Base of the branch.

    Returns:
    SegP (list): Modified parent segment.
    Base (np.ndarray): Expanded base of the branch.
    """

    
    Base = SegC[0].copy()
    if Base.size > 0:
        # Define the directions of the segments
        DirChi = segment_direction(Ce, SegC, 0)
        DirPar = segment_direction(Ce, SegP, nl)
        if len(Base) > 1:
            BaseCent = np.mean(Ce[Base.astype(int), :], axis=0)
            db, _v,_h,_b = distances_to_line(Ce[Base.astype(int), :], DirChi, BaseCent)  # Distances of the sets in the base to the axis of the branch
            DiamBase = 2 * np.max(db)  # Diameter of the base
        elif len(Bal[Base[0]]) > 1:
            BaseCent = np.mean(P[Bal[Base[0]], :], axis=0)
            db, _v,_h,_b = distances_to_line(P[Bal[Base[0]], :], DirChi, BaseCent)
            DiamBase = 2 * np.max(db)
        else:
            BaseCent = Ce[Base[0], :]
            DiamBase = 0

        # Determine the number of cover set layers "n" to be checked
        Angle = abs(np.dot(DirChi, DirPar))  # Abs of cosine of the angle between component and segment directions
        Nlayer = max(3, int(np.ceil(Angle * 2 * DiamBase / PatchDiam)))
        if Nlayer > nl+1:  # Can go only to the bottom of the segment
            Nlayer = int(nl)+1 


        layer = 0
        base[0] = Base
        nl = int(nl)
        while layer < Nlayer:
            Sets = SegP[nl - layer].copy()  
            Seg = np.mean(Ce[Sets.astype(int), :], axis=0)  # Mean of the cover sets' centers

            VBase = mat_vec_subtraction(Ce[Sets.astype(int), :], BaseCent)  # Vectors from base's center to sets in the segment
            h = np.dot(VBase, DirChi)
            B = np.outer(h, DirChi)
            V = VBase - B
            distSets = np.sqrt(np.sum(V**2, axis=1))  # Distances of the sets in the segment to the axis of the branch

            VSeg = mat_vec_subtraction(Ce[Sets.astype(int), :], Seg)  # Vectors from segment's center to sets in the segment
            lenBase = np.sqrt(np.sum(VBase**2, axis=1))  # Lengths of VBase
            lenSeg = np.sqrt(np.sum(VSeg**2, axis=1))  # Lengths of VSeg

            if Angle < 0.9:
                K = lenBase < 1.1 / (1 - 0.5 * Angle**2) * lenSeg  # Sets closer to the base's center than segment's center
                J = distSets < 1.25 * DiamBase  # Sets close enough to the axis of the branch
                I = K & J
            else:  # Branch almost parallel to parent
                I = distSets < 1.25 * DiamBase  # Only the distance to the branch axis counts

            if np.all(I) or not np.any(I):  # Stop the process if all the segment's or no segment's sets
                layer = Nlayer+1
            else:
                SegP[nl - layer ] = Sets[~I].copy()
                base[layer + 1] = Sets[I].copy()
                layer += 1

        Base = np.hstack([b for b in base[:Nlayer +2] if b is not None])


    return SegP, Base


def segment_direction(Ce, Seg, nl):
    """
    Defines the direction of the segment.

    Parameters:
    Ce (np.ndarray): Centers of the cover sets.
    Seg (list): Segment.
    nl (int): Layer index in the segment.

    Returns:
    D (np.ndarray): Direction vector of the segment.
    """



    # Define bottom and top layers
    if nl - 3 > -1:
        bot = nl - 3
    else:
        bot = 0  

    j = 0
    while j < 2 and len(Seg[int(bot)]) == 0:  # Check for empty layers
        bot += 1
        j += 1

    if nl + 2 < len(Seg):
        top = nl + 2
    else:
        top = len(Seg) - 1  

    j = 0
    while j < 2 and len(Seg[int(top)]) == 0:  # Check for empty layers
        top -= 1
        j += 1

    # Direction
    if top > bot:
        #Original treeQSM uses own average function, confirm this is the same math
        Bot = np.mean(Ce[Seg[int(bot)].astype(int), :],axis=0)
        Top = np.mean(Ce[Seg[int(top)].astype(int), :],axis=0)
        V = Top - Bot
        D = V / np.linalg.norm(V)
    else:
        D = np.zeros(3)

    return D