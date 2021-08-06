# edges = {   
#             (0, 4): 7,
#             (0, 5): 8,
#             (1, 5): 11,
#             (1, 6): 15,
#             (1, 8): 9,
#             (2, 4): 4,
#             (2, 7): 14,
#             (2, 8): 3,
#             (3, 6): 12,
#             (3, 7): 13,
#             (4, 5): 6,
#             (4, 8): 5,
#             (5, 8): 10,
#             (6, 7): 0,
#             (6, 8): 2,
#             (7, 8): 1 
#         }
# nodes = ['A', 'C', 'G', 'I', 'D', 'B', 'F', 'H', 'E']
edges = {(0, 5): 10,
 (0, 6): 11,
 (1, 7): 23,
 (1, 8): 31,
 (1, 14): 21,
 (2, 4): 4,
 (2, 11): 29,
 (2, 13): 3,
 (3, 9): 27,
 (3, 10): 28,
 (4, 5): 13,
 (4, 12): 12,
 (4, 13): 5,
 (5, 6): 9,
 (5, 12): 14,
 (6, 7): 17,
 (6, 12): 16,
 (7, 12): 15,
 (7, 14): 22,
 (8, 9): 32,
 (8, 14): 26,
 (8, 15): 24,
 (9, 10): 0,
 (9, 15): 2,
 (10, 11): 30,
 (10, 15): 1,
 (11, 13): 7,
 (11, 15): 6,
 (12, 13): 19,
 (12, 14): 20,
 (13, 14): 18,
 (13, 15): 8,
 (14, 15): 25}
nodes = ['A', 'D', 'M', 'P', 'I', 'E', 'B', 'C', 'H', 'L', 'O', 'N', 'F', 'J', 'G', 'K']

edges = sorted(edges.items(), key=lambda x: x[1])

# print(edges)
k = []
for edge, edge_id in edges:

    e = nodes[edge[0]] + nodes[edge[1]]
    k.append(e)
print(k)