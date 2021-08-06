import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify
import pdb


class MeshPool(nn.Module):
    
    def __init__(self, target, multi_thread=False):
        print(f'target edges: {target}')
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        # for each mesh call __pool_main
        # to do it parallely create threads and collect result.
        # fe.shape -> [1, 16, 16]
        self.__updated_fe = [[] for _ in range(len(meshes))] # len(meshes) -> 1
        # self.__updated_fe = [[]]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1, self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        # build a priority queue for the mesh edges
        # check - http://staff.ustc.edu.cn/~csli/graduate/algorithms/book6/chap07.htm for details
        # lowest mag edges at top of heap
        # these edges get removed first
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.edges_count], mesh.edges_count)
        # recycle = []
        # last_queue_len = len(queue)
        last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        print(self.__out_target)
        # pdb.set_trace()
        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            
            # print(edge_id)
            edge_id = int(edge_id)
            # if edge_id in [8, 19, 20, 25]:
            # if edge_id == 19:
            #     pdb.set_trace()
            # else:
            #     continue
            if mask[edge_id]:
                self.__pool_edge(mesh, edge_id, mask, edge_groups)
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):
        if self.has_boundaries(mesh, edge_id):
            # if its a boundary edge like [A, D] you cannot pool it
            return False
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0)\
            and self.__clean_side(mesh, edge_id, mask, edge_groups, 2) \
            and self.__is_one_ring_valid(mesh, edge_id):
            self.__merge_edges[0] = self.__pool_side(mesh, edge_id, mask, edge_groups, 0) # edge_id=19, mask -> all ones
            # redirected gemm edges from edges 18 and 19, deleted edge 18, total edges = 32
            # self.__merge_edges[0] = 20
            self.__merge_edges[1] = self.__pool_side(mesh, edge_id, mask, edge_groups, 2) # edge_id=19, mask -> all ones
            
            # self.__merge_edges = [18, 12]
            # self.__merge_edges = [20, 5]
            mesh.merge_vertices(edge_id) # edge_id -> 19
            mask[edge_id] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_id)
            mesh.edges_count -= 1
            return True
        else:
            return False

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            # if your number of edges are less than what you intend to have after pooling
            # then you do not need to pool the edge
            return False

       
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups, side)
       
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups, side)
        return True

    @staticmethod
    def has_boundaries(mesh, edge_id):
        # if edge is like the [A, D] edge
        # if any of your neighbors are boundary edges like [A, D] you cannot pool that edge
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False


    @staticmethod
    def __is_one_ring_valid(mesh, edge_id):
        # edge_id = 19, 4 nbrs - [15, 8, 5, 12]
        # v_a = vertices of edge_id [12, 13] FJ
        v_a = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1)) # set of edge ids 12 belongs to - length is 6
        v_b = set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1)) # set of edge ids 13 belongs to - length is 6
        # v_a = {4, 5, 6, 7, 12, 13, 14}
        # v_b = {2, 4, 11, 12, 13, 14, 15}
        # v_a & v_b - {13, 4, 12, 14}
        # {13, 4, 12, 14}  - {12, 13} = {4, 14}
        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 2

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        # (20, 18, 1, 0, 2, 2, [15, 22], [25, 8])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2, other_keys_b[0], mesh.sides[key_b, other_side_b])
        # mesh.gemm_edges[20] = [18, 19, 15, 22] -> [25, 19, 15, 22] # 18 replaced with 25
        # mesh.gemm_edges[25] = [26, 24,  8, 18] -> [26, 24,  8, 20] # 18 replaced with 20
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1, other_keys_b[1], mesh.sides[key_b, other_side_b + 1])
        # mesh.gemm_edges[20] = [25, 19, 15, 22] -> [25,  8, 15, 22] # 19 replaced by 8
        # mesh.gemm_edges[8] = [ 6,  7, 18, 25] -> [ 6,  7, 20, 25] # 18 replaced by 20
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a) # groups[20,:] = zeros except at 20 and 18
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a) # groups[20,:] = zeros except at 20, 18 and 19
        mask[key_b] = False # mask[18] = False, 2nd pass key_b = 12
        # redirect all edges from 18 and 12 and remove edge 18 and 12
        MeshPool.__remove_group(mesh, edge_groups, key_b) # no changes initially as there is no history object
        mesh.remove_edge(key_b) # remove edge_id association with vertices of that edge.
        mesh.edges_count -= 1 # decrease overall count of edges

        # second time
        # (5, 12, 3, 2, 0, 0, [3, 4], [13, 14])
        # mesh.gemm_edges[5] -> [ 3,  4, 12, 19] -> [ 3,  4, 13, 19] # 12 replaced by 13
        # mesh.gemm_edges[13] -> [14, 12, -1, -1] -> [14,  5, -1, -1] # 12 replaced by 5
        # mesh.gemm_edges[5] -> [ 3,  4, 13, 19] -> [ 3,  4, 13, 14] # 19 replaced  by 14
        # mesh.gemm_edges[14] -> [12, 13,  9, 16] -> [ 5, 13,  9, 16] $ 12 replaced with 5
        # union groups -> groups[5, :] = all zeros except 5, 12
        # union groups -> groups[5, :] = all zeros except 5, 12, 19
        # mask[12] -> False
        # remove edge with edge_id 12
        return key_a, # final_return key_a = 5

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b = info
        # main_edge = 19, key_a = 20, key_b = 18 // FJ, JG, FG
        # mesh.sides[edge_id] or mesh.sides[19] = [1, 0, 3, 2] // [key_b, key_a, nbr_other_a, nbr_other_b]
        # side_a = 1, side_b = 0, other_side_a = 2, other_side_b = 2
        # other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        # triangles supported by face nbrs are accounted
        # other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        # other_keys_a = [15, 22], other_keys_b = [25, 8] // (CF, CG), (GK, JK)
        
        shared_items = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        # if there are no shared items in other_keys_a and other_keys_b then its valid
        # 
        if len(shared_items) == 0:
            return []
        else:
            # ignoring this condition as it doesnt occur in my example
            # it would occur for case 2 kind of edges 
            assert (len(shared_items) == 2)
            # let GK be shared, then other_keys_a = [15, 25], other_keys_b = [25, 8], shared_items = [1, 0]
            middle_edge = other_keys_a[shared_items[0]] # middle_edge = 25
            update_key_a = other_keys_a[1 - shared_items[0]] # update_key_a = 15
            update_key_b = other_keys_b[1 - shared_items[1]] # update_key_b = 8
            update_side_a = mesh.sides[key_a, other_side_a + 1 - shared_items[0]] # update_side_a = 3
            update_side_b = mesh.sides[key_b, other_side_b + 1 - shared_items[1]] # update_side_b = 2
            # redirect_edges(FJ, CF, sides) here 
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a, update_side_a)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b, update_side_b) # 19, 1, 8, 2 
            MeshPool.__redirect_edges(mesh, update_key_a, MeshPool.__get_other_side(update_side_a), update_key_b, MeshPool.__get_other_side(update_side_b))
            # 15, 2, 8, 3
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id) # zeros(33x33), 20, 19, # groups[19,:] = all zeros except at position 19, 20
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id) # (33x33), 18, 19,  groups[19, :] = all zeros except at 18, 19, 20 position
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a) # groups[15,:] = all zeros except at position 20, 15
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_a) # groups[15,:] = all zeros except at position 20, 15, 25
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b) # groups[8,:] = all zeros except at position 18, 8
            MeshPool.__union_groups(mesh, edge_groups, middle_edge, update_key_b) # groups[8,:] = all zeros except at position 18, 8, 25
            return [key_a, key_b, middle_edge] # 20, 18, 25, (GF, GJ, GK) edges are considered invalid

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        # 19 - FJ, side_a = 0, CF - edge_b_key = 15, side_b = 3
        # 19 - gemm edges [20, 18, 5, 12]  # FG, JG, IJ, IF
        # 15 - gemm edges [16, 17, 22, 20] # BF, BC, CG, FG
        # mesh.sides[19] - [1,0,3,2], mesh.sides[15] - [1,0,3,2]
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key # replace FG with FK
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key # replace FG with FJ
        mesh.sides[edge_a_key, side_a] = side_b # mesh.sides[19, 0] = 3
        mesh.sides[edge_b_key, side_b] = side_a # mesh.sides[15, 3] = 0
        # mesh.gemm_edges[19] = [20, 18, 5, 12] -> [15, 18, 5, 12]
        # mesh.gemm_edges[15] = [16, 17, 22, 20] -> [16, 17, 22, 19]

        # second time
        # mesh.gemm_edges[19] = [15, 18, 5, 12] -> [15, 8, 5, 12]
        # mesh.gemm_edges[8] = [6, 7, 18, 25] -> [6, 7, 19, 25]
        # 19 - FJ, side_a = 1, side_b = 2, JK edge_b_key = 8, gemm edges a - FK, JK, IJ, IF
        # gemm edges b - NK, NJ, JG, GK
        # 19 - gemm edges [15,  8,  5, 12] # replace 8 with 8
        # 8 - gemm edges [ 6,  7, 18, 25] # replace 18 with 19 , replace JG with FJ

        # third time
        # 15, 2, 8, 3
        # mesh.gemm_edges[15] = [16, 17, 22, 19] -> [16, 17, 8, 19]
        # mesh.gemm_edges[8] = [6, 7, 19, 25] -> [6,7, 19, 15] 
        # in total final gemm_edges are as follows
        # mesh.gemm_edges[19] = [15, 8, 5, 12]
        # mesh.gemm_edges[15] = [16, 17, 8, 19]
        # mesh.gemm_edges[8] =  [6, 7, 19, 15]
        

    @staticmethod
    def __get_shared_items(list_a, list_b):
        # [1, 30], [29, 3]
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2) 
        # oside for updated side a -> 3 + 1 - 2* (3 % 2) = 2
        # oside for updated side b -> 2 + 1 -  2*(2%2) = 3

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        '''
        key_a - edge_id of sister edge, JG
        key_b - edge_id of sister edge, GK
        Sides tell the order in which the edge neighbors are placed
        side_a - 
        
        '''
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        other_side_a = (side_a - (side_a % 2) + 2) % 4
        other_side_b = (side_b - (side_b % 2) + 2) % 4 
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a], mesh.gemm_edges[key_a, other_side_a + 1]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b], mesh.gemm_edges[key_b, other_side_b + 1]]
        return key_a, key_b, side_a, side_b, other_side_a, other_side_b, other_keys_a, other_keys_b

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert(len(vertex) == 1)
        mesh.remove_vertex(vertex[0])

    def __build_queue(self, features, edges_count):
        # features - [16, 16]
        # create a heap based on unsqared L2 norm of features of each edge
        # we do this because we need to delete the edges with the smallest unsquared L2 norm
        squared_magnitude = torch.sum(features * features, 0) # shape -> [16,]

        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1) # shape -> [16, 1]
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device, dtype=torch.float32).unsqueeze(-1) # 16 x 1
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist() 
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target) # zeros(33x33), 20, 19
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)

