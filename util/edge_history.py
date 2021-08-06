import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl
import numpy as np
import pdb
V = np.array
r2h = lambda x: colors.rgb2hex(tuple(map(lambda y: y / 255., x)))
surface_color = r2h((255, 230, 205))
edge_color = r2h((90, 90, 90))
edge_colors = (r2h((15, 167, 175)), r2h((230, 81, 81)), r2h((142, 105, 252)), r2h((248, 235, 57)),
               r2h((51, 159, 255)), r2h((225, 117, 231)), r2h((97, 243, 185)), r2h((161, 183, 196)))


selected_edge_colors = (r2h((0, 255, 0)), r2h((255, 0, 0)), r2h((0, 0, 255)))

def init_plot():
    fig = pl.figure()
    fig.set_size_inches(8, 6)
    ax = fig.add_subplot(111, projection='3d')
    # hide axis, thank to
    # https://stackoverflow.com/questions/29041326/3d-plot-with-matplotlib-hide-axes-but-keep-axis-labels/
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    return (ax, [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf], fig)


def update_lim(mesh, plot):
    vs = mesh[0]
    for i in range(3):
        plot[1][2 * i] = min(plot[1][2 * i], vs[:, i].min())
        plot[1][2 * i + 1] = max(plot[1][2 * i], vs[:, i].max())
    return plot


def update_plot(mesh, plot):
    if plot is None:
        plot = init_plot()
    return update_lim(mesh, plot)


def surfaces(mesh, plot):
    vs, faces, edges = mesh
    vtx = vs[faces]
    edgecolor = edge_color if not len(edges) else 'none'
    tri = a3.art3d.Poly3DCollection(vtx, facecolors=surface_color +'55', edgecolors=edgecolor,
                                    linewidths=.5, linestyles='dashdot')
    plot[0].add_collection3d(tri)
    return plot


def segments(mesh, plot):
    vs, _, edges = mesh
    for edge_c, edge_group in enumerate(edges):
        # print(edge_group.shape)
        for edge_idx in edge_group:
            edge = vs[edge_idx]
            line = a3.art3d.Line3DCollection([edge],  linewidths=.5, linestyles='dashdot')
            line.set_color(selected_edge_colors[edge_c % len(selected_edge_colors)])
            plot[0].add_collection3d(line)
    return plot


def plot_mesh(mesh, *whats, show=True, plot=None):
    for what in [update_plot] + list(whats):
        plot = what(mesh, plot)
    if show:
        li = max(plot[1][1], plot[1][3], plot[1][5])
        plot[0].auto_scale_xyz([0, li], [0, li], [0, li])
        pl.tight_layout()
        pl.show()
        # pdb.set_trace()
        plot[2].savefig('./temp.png')
    return plot


def parse_obje(obj_file, highlighted_edges_file, scale_by):
    vs = []
    faces = []
    edges = []

    def add_to_edges():
        if edge_c >= len(edges):
            for _ in range(len(edges), edge_c + 1):
                edges.append([])
        edges[edge_c].append(edge_v)

    def fix_vertices():
        nonlocal vs, scale_by
        vs = V(vs)
        z = vs[:, 2].copy()
        vs[:, 2] = vs[:, 1]
        vs[:, 1] = z
        max_range = 0
        for i in range(3):
            min_value = np.min(vs[:, i])
            max_value = np.max(vs[:, i])
            max_range = max(max_range, max_value - min_value)
            vs[:, i] -= min_value
        if not scale_by:
            scale_by = max_range
        vs /= scale_by

    with open(highlighted_edges_file) as f:
        selected_set = set()
        for line in f:
            line = tuple((float(c) for c in line.strip().split(' ')))
            selected_set.add(line)

    # print(selected_set)

    with open(obj_file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:]])
            elif splitted_line[0] == 'f':
                faces.append([int(c) - 1 for c in splitted_line[1:]])
            elif splitted_line[0] == 'e':
                print([int(c) - 1 for c in splitted_line[1:-1]])
                # if len(splitted_line) < 4 and selected_set is not None:
                #     if 
                #     splitted_line.append()
                if len(splitted_line) >= 4:
                    edge_v = [int(c) - 1 for c in splitted_line[1:-1]]
                    print(edge_v)
                    edge_c = int(splitted_line[-1])
                    print(edge_c)
                    add_to_edges()
                

    vs = V(vs)
    fix_vertices()
    faces = V(faces, dtype=int)
    edges = [V(c, dtype=int) for c in edges]
    return (vs, faces, edges), scale_by

def parse_mesh(mesh, highlighted_edges_file, scale_by):
    # should return vs, faces, edges and scale_by factor
    # vs: np.ndarray (float)
    # faces: np.ndarray (int)
    # edges ?
    # all edges belonging to highlighted edges file belong to one group
    # all others belong to another group
    new_edges = [[], []]# 2 groups - selected, not-selected
    new_edges = [[], [], []] # selected, not selected, non-manifold
    
    with open(highlighted_edges_file) as f:
        selected_set = set()
        for line in f:
            line = tuple((float(c) for c in line.strip().split(' ')))
            selected_set.add(line)

    def fix_vertices():
        nonlocal vs, scale_by
        vs = V(vs)
        z = vs[:, 2].copy()
        vs[:, 2] = vs[:, 1]
        vs[:, 1] = z
        max_range = 0
        for i in range(3):
            min_value = np.min(vs[:, i])
            max_value = np.max(vs[:, i])
            max_range = max(max_range, max_value - min_value)
            vs[:, i] -= min_value
        if not scale_by:
            scale_by = max_range
        vs /= scale_by

    
    vs, faces, edges = convert_mesh(mesh)
    non_selected_edge_color = (0,255,0)
    selected_edge_color = (255, 0, 0)
    for edge in edges:
        nodeA, nodeB = edge
        if tuple(mesh.vs[nodeA]) in selected_set or tuple(mesh.vs[nodeB]) in selected_set:
            new_edges[0].append(edge)
        else:
            new_edges[1].append(edge)
    # pdb.set_trace()
    fix_vertices()
    # face is 4 vertices not 4 edge ids!    
    return (np.array(vs), np.array(faces), np.array(new_edges)), scale_by
    # pdb.set_trace() 
    


def view_meshes(*files, offset=.2):
    plot = None
    max_x = 0
    scale = 0
    for file, highlighted_edges_file in files:
        # mesh, scale = parse_obje(file, highlighted_edges_file, scale)
        mesh, scale = parse_mesh(file, highlighted_edges_file, scale)
        max_x_current = mesh[0][:, 0].max()
        mesh[0][:, 0] += max_x + offset
        # pdb.set_trace()

        plot = plot_mesh(mesh, surfaces, segments, plot=plot, show=file == files[-1])
        # azims = [-60, -30, 0, 30, 60]
        # dists = [5, 10, 15]
        # elevs = [-30, 0, 10, 30]
        azims = [-60, 0, 30]
        dists = [0]
        elevs = [0, 10, 30]
        
        import itertools
        for i, (azim, dist, elev) in enumerate(itertools.product(azims, dists, elevs)):
            plot[0].azim = azim
            plot[0].dist = dist
            plot[0].elev = elev
            plot[2].savefig(f'./temp1/temp_{i}.png', dpi=300)
            # 27 -> (0, 5, 30), 38 -> (30, 5, 10), 39-> (30, 5, 30), 42 -> (30, 10, 10)
            print(i, (azim, dist, elev))
        pdb.set_trace()
        max_x += max_x_current + offset


def convert_mesh(mesh):

    faces = []
    final_vs = []
    final_faces = []
    final_edges = []
    pdb.set_trace()
    vs = mesh.vs[mesh.v_mask]
    gemm = np.array(mesh.gemm_edges)
    new_indices = np.zeros(mesh.v_mask.shape[0], dtype=np.int32)
    new_indices[mesh.v_mask] = np.arange(0, np.ma.where(mesh.v_mask)[0].shape[0])

    for edge_index in range(len(gemm)):
        cycles = mesh.get_cycle(gemm, edge_index)
        for cycle in cycles:
            faces.append(mesh.cycle_to_face(cycle, new_indices))

    for v in vs:
        final_vs.append([v[0], v[1], v[2]])
        # vcol = ' %f %f %f' % (vcolor[vi, 0], vcolor[vi, 1], vcolor[vi, 2]) if vcolor is not None else ''
        # f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], vcol))
    for face_id in range(len(faces) - 1):
        # f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
        final_faces.append([faces[face_id][0], faces[face_id][1], faces[face_id][2]])
    # f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
    final_faces.append([faces[-1][0], faces[-1][1], faces[-1][2]])
    for edge in mesh.edges:
        # f.write("\ne %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1))
        final_edges.append([new_indices[edge[0]], new_indices[edge[1]]])
    return final_vs, final_faces, final_edges