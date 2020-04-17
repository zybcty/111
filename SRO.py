#  -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pyvoro
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
import re
import math
import pandas as pd
import os


def mkdir(path_write):
    # 判断目录是否存在
    # 存在：True
    # 不存在：False
    folder = os.path.exists(path_write)
    # 判断结果
    if not folder:
        # 如果不存在，则创建新目录
        os.makedirs(path_write)
        print('-----创建成功-----')
    else:
        # 如果目录已存在，则不创建，提示目录已存在
        print('目录已存在')


def compute_simplice_area(vertice1, vertice2, vertice3):
    # problem1: compute error -> eij = eik causes error.
    # neglect
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    h = (np.dot(eij, eij) - (np.dot(eij, eik) / (np.linalg.norm(eik))) ** 2) ** 0.5
    return (np.linalg.norm(eik)) * h / 2


def compute_area(vertices_input, adjacent_cell_input, vertices_id_input, simplice_input):
    area_judge_in = np.zeros(shape=[len(adjacent_cell_input), ], dtype=int)
    area_in = np.zeros(shape=[len(adjacent_cell_input), ])
    sing_area = np.zeros(shape=[len(simplice_input), ])
    for a in range(len(simplice_input)):
        sing_area[a] = compute_simplice_area(vertices_input[simplice_input[a][0]],
                                             vertices_input[simplice_input[a][1]],
                                             vertices_input[simplice_input[a][2]])
    for a in range(len(simplice_input)):
        for b in range(len(adjacent_cell_input)):
            if simplice_input[a][0] in vertices_id_input[b]:
                if simplice_input[a][1] in vertices_id_input[b]:
                    if simplice_input[a][2] in vertices_id_input[b]:
                        area_in[b] += sing_area[a]
    average_area = np.mean(area_in)
    for a in range(len(adjacent_cell_input)):
        if area_in[a] >= 0.05 * average_area:
            area_judge_in[a] = 1
    return area_judge_in, area_in


def compute_cluster_packing_efficiency(voroni_neighbour_use_input, points_input, radius_input):
    cluster_packing_efficiency = []
    for a in range(len(voroni_neighbour_use_input)):
        if len(voroni_neighbour_use_input[a]) >= 4:
            points_now = []
            radius_now = []
            origin_particle = points_input[a]
            origin_radius = radius_input[a]
            for b in range(len(voroni_neighbour_use_input[a])):
                points_now.append(points_input[voroni_neighbour_use_input[a][b]])
                radius_now.append(radius_input[voroni_neighbour_use_input[a][b]])
            cpe_ch = ConvexHull(points_now)
            cpe_simplice = np.array(cpe_ch.simplices)
            interstice_volume_mid = np.zeros(shape=[len(cpe_simplice), ])
            cluster_packing_efficiency_x = compute_cluster_packing_efficiency_single_particle(cpe_simplice,
                                                                                              np.array(points_now),
                                                                                              np.array(radius_now),
                                                                                              interstice_volume_mid,
                                                                                              origin_particle,
                                                                                              origin_radius)

            cluster_packing_efficiency.append(cluster_packing_efficiency_x)
        else:
            cluster_packing_efficiency.append(0.0)
    return cluster_packing_efficiency


def compute_cluster_packing_efficiency_single_particle(simplice_input, points_now, radius_now, interstice_volume_mid,
                                                       origin_particle, origin_radius):
    triangle_volume_x = np.zeros_like(interstice_volume_mid)
    pack_volume_x = np.zeros_like(interstice_volume_mid)
    for b in range(len(simplice_input)):
        volume_triangle = compute_tetrahedron_volume(points_now[simplice_input[b][0]],
                                                     points_now[simplice_input[b][1]],
                                                     points_now[simplice_input[b][2]],
                                                     origin_particle)
        volume_pack = (compute_solide_angle(origin_particle, points_now[simplice_input[b][0]],
                                            points_now[simplice_input[b][1]],
                                            points_now[simplice_input[b][2]])
                       * origin_radius ** 3 +
                       compute_solide_angle(points_now[simplice_input[b][2]], origin_particle,
                                            points_now[simplice_input[b][0]], points_now[simplice_input[b][1]])
                       * radius_now[simplice_input[b][2]] ** 3 +
                       compute_solide_angle(points_now[simplice_input[b][1]],
                                            points_now[simplice_input[b][2]], origin_particle,
                                            points_now[simplice_input[b][0]])
                       * radius_now[simplice_input[b][1]] ** 3 +
                       compute_solide_angle(points_now[simplice_input[b][0]], points_now[simplice_input[b][1]],
                                            points_now[simplice_input[b][2]], origin_particle) * radius_now[
                           simplice_input[b][0]] ** 3) / 3
        triangle_volume_x[b] = volume_triangle
        pack_volume_x[b] = volume_pack
    return np.sum(pack_volume_x) / np.sum(triangle_volume_x)


def compute_tetrahedron_volume(vertice1, vertice2, vertice3, vertice4):
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    eil = np.array([vertice4[0] - vertice1[0], vertice4[1] - vertice1[1], vertice4[2] - vertice1[2]])
    return abs(np.dot(eil, np.cross(eij, eik))) / 6


def compute_solide_angle(vertice1, vertice2, vertice3, vertice4):
    eij = np.array([vertice2[0] - vertice1[0], vertice2[1] - vertice1[1], vertice2[2] - vertice1[2]])
    eik = np.array([vertice3[0] - vertice1[0], vertice3[1] - vertice1[1], vertice3[2] - vertice1[2]])
    eil = np.array([vertice4[0] - vertice1[0], vertice4[1] - vertice1[1], vertice4[2] - vertice1[2]])
    len_eij = np.linalg.norm(eij)
    len_eik = np.linalg.norm(eik)
    len_eil = np.linalg.norm(eil)
    return 2 * math.atan2(math.fabs(np.dot(eij, np.cross(eik, eil))),
                          (len_eij * len_eik * len_eil + np.dot(eij, eik) * len_eil
                           + np.dot(eij, eil) * len_eik + np.dot(eik, eil) * len_eij))


def compute_coordination_number_by_cutoff_distance(points_input, radius_input):
    maxdistance = 3.0 * radius_input[0]
    kdtree = KDTree(points_input)
    pairs = list(kdtree.query_pairs(maxdistance))
    cutoff_bonds = []
    for a in range(len(points_input)):
        cutoff_bonds.append([])
    for a in range(len(pairs)):
        cutoff_bonds[pairs[a][0]].append(pairs[a][1])
        cutoff_bonds[pairs[a][1]].append(pairs[a][0])
    coordination_number_by_cutoff_distance_in = []
    for a in range(len(cutoff_bonds)):
        coordination_number_by_cutoff_distance_in.append(len(cutoff_bonds[a]))
    return coordination_number_by_cutoff_distance_in


def compute_cellfraction(voro_input, radius_input):
    volume = []
    for a in range(len(voro_input)):
        volume.append(voro_input[a]['volume'])
    ball_volume = (np.max(radius_input) ** 3) * 4 * math.pi / 3
    cellfraction_in = [ball_volume / volume[a] for a in range(len(voro_input))]
    return cellfraction_in


def compute_weight_i_fold_symm(voro_input, area_all_input):
    area_weight_i_fold_symm3_in, area_weight_i_fold_symm4_in, area_weight_i_fold_symm5_in, \
    area_weight_i_fold_symm6_in, area_weight_i_fold_symm7_in = [], [], [], [], []
    for a in range(len(voro_input)):
        faces_in = voro_input[a]['faces']
        vertices_id_length = []
        for b in range(len(faces_in)):
            vertices_id_length.append(len(faces_in[b]['vertices']))
        id3 = [c for c, b in enumerate(vertices_id_length) if b == 3]
        id4 = [c for c, b in enumerate(vertices_id_length) if b == 4]
        id5 = [c for c, b in enumerate(vertices_id_length) if b == 5]
        id6 = [c for c, b in enumerate(vertices_id_length) if b == 6]
        id7 = [c for c, b in enumerate(vertices_id_length) if b == 7]
        area3, area4, area5, area6, area7 = 0, 0, 0, 0, 0
        for b in range(len(id3)):
            area3 += area_all_input[a][id3[b]]
        for b in range(len(id4)):
            area4 += area_all_input[a][id4[b]]
        for b in range(len(id5)):
            area5 += area_all_input[a][id5[b]]
        for b in range(len(id6)):
            area6 += area_all_input[a][id6[b]]
        for b in range(len(id7)):
            area7 += area_all_input[a][id7[b]]
        area_total = area3 + area4 + area5 + area6 + area7
        area_weight_i_fold_symm3_in.append(area3 / area_total)
        area_weight_i_fold_symm4_in.append(area4 / area_total)
        area_weight_i_fold_symm5_in.append(area5 / area_total)
        area_weight_i_fold_symm6_in.append(area6 / area_total)
        area_weight_i_fold_symm7_in.append(area7 / area_total)
    return area_weight_i_fold_symm3_in, area_weight_i_fold_symm4_in, area_weight_i_fold_symm5_in, \
           area_weight_i_fold_symm6_in, area_weight_i_fold_symm7_in


def compute_voroni_idx(voro_input):
    voroni_idx3_in, voroni_idx4_in, voroni_idx5_in, voroni_idx6_in, voroni_idx7_in = [], [], [], [], []
    i_fold_symm3_in, i_fold_symm4_in, i_fold_symm5_in, i_fold_symm6_in, i_fold_symm7_in = [], [], [], [], []
    for a in range(len(voro_input)):
        faces_in = voro_input[a]['faces']
        vertices_id_length = []
        for b in range(len(faces_in)):
            vertices_id_length.append(len(faces_in[b]['vertices']))
        count_all = vertices_id_length.count(3) + vertices_id_length.count(4) \
                    + vertices_id_length.count(5) + vertices_id_length.count(6) \
                    + vertices_id_length.count(7)
        voroni_idx3_in.append(vertices_id_length.count(3))
        voroni_idx4_in.append(vertices_id_length.count(4))
        voroni_idx5_in.append(vertices_id_length.count(5))
        voroni_idx6_in.append(vertices_id_length.count(6))
        voroni_idx7_in.append(vertices_id_length.count(7))
        i_fold_symm3_in.append(vertices_id_length.count(3) / count_all)
        i_fold_symm4_in.append(vertices_id_length.count(4) / count_all)
        i_fold_symm5_in.append(vertices_id_length.count(5) / count_all)
        i_fold_symm6_in.append(vertices_id_length.count(6) / count_all)
        i_fold_symm7_in.append(vertices_id_length.count(7) / count_all)
    return voroni_idx3_in, voroni_idx4_in, voroni_idx5_in, voroni_idx6_in, voroni_idx7_in, \
           i_fold_symm3_in, i_fold_symm4_in, i_fold_symm5_in, i_fold_symm6_in, i_fold_symm7_in


path = 'cyc5300fric01shearrate025'
path_output = '../' + path + '/some old feature SRO'
mkdir(path_output)
initial_step = 15400000
xmin, xmax = -0.20, 0.20
ymin, ymax = -0.20, 0.20
zmin, zmax = -0.12, 0.12
interval_step = 800000
cyc_number = 1250
ini_number = 750


for i in range(cyc_number):
    i = i + ini_number
    print i
    step = initial_step + i * interval_step
    tem = '%d' % step
    tem1 = '%d' % i
    filename = 'dump-' + tem + '.sample'

    atomfile = open('../' + path + '/sort position/' + filename, 'r')
    lines = atomfile.readlines()
    atomfile.close()
    lines = lines[9:]
    particle_id = map(int, map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[0] for line in lines]))
    position_x = map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[3] for line in lines])
    position_y = map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[4] for line in lines])
    position_z = map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[5] for line in lines])
    radius = map(float, [re.findall(r'-?\d+\.?\d*e?[-+]?\d*', line)[2] for line in lines])
    points = np.array(zip(position_x, position_y, position_z))
    zmax = np.max(position_z) + np.max(radius)
    limits = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
    dispersion = 5.0 * np.max(radius)
    voro = pyvoro.compute_voronoi(points, limits, dispersion, periodic=[False] * 3)

    adjacent_cell_all = []
    area_all = []
    for x in range(len(voro)):
        vertices = voro[x]['vertices']
        ch = ConvexHull(vertices)
        simplice = np.array(ch.simplices)
        faces = voro[x]['faces']
        adjacent_cell = []
        for y in range(len(faces)):
            adjacent_cell.append(faces[y]['adjacent_cell'])
        vert_id = []
        for y in range(len(faces)):
            vert_id.append(faces[y]['vertices'])
        area_judge, area = compute_area(vertices, adjacent_cell, vert_id, simplice)
        adjacent_cell_use = []
        for y in range(len(adjacent_cell)):
            if area_judge[y] == 1:
                adjacent_cell_use.append(adjacent_cell[y])
        adjacent_cell_all.append(adjacent_cell_use)
        area_all.append(area)

    voroni_neighbour = []
    for x in range(len(adjacent_cell_all)):
        voroni_neighbour_now = []
        for value in adjacent_cell_all[x]:
            if value >= 0:
                voroni_neighbour_now.append(value)
        voroni_neighbour.append(voroni_neighbour_now)
    bonds = []
    for x in range(len(voroni_neighbour)):
        for y in range(len(voroni_neighbour[x])):
            if voroni_neighbour[x][y] > x:
                bonds.append([x, voroni_neighbour[x][y]])
    bonds = np.array(bonds)
    voroni_neighbour_use = []
    for x in range(len(lines)):
        voroni_neighbour_use.append([])
    for x in range(len(bonds)):
        voroni_neighbour_use[bonds[x][0]].append(bonds[x][1])
        voroni_neighbour_use[bonds[x][1]].append(bonds[x][0])

    Coordination_number_by_Voroni_tessellation = []
    for x in range(len(voroni_neighbour_use)):
        Coordination_number_by_Voroni_tessellation.append(len(voroni_neighbour_use[x]))

    area_weight_i_fold_symm3, area_weight_i_fold_symm4, area_weight_i_fold_symm5, \
    area_weight_i_fold_symm6, area_weight_i_fold_symm7 = compute_weight_i_fold_symm(voro, area_all)

    Coordination_number_by_cutoff_distance = compute_coordination_number_by_cutoff_distance(points, radius)

    cellfraction = compute_cellfraction(voro, radius)

    Voroni_idx3, Voroni_idx4, Voroni_idx5, Voroni_idx6, Voroni_idx7, \
    i_fold_symm3, i_fold_symm4, i_fold_symm5, i_fold_symm6, i_fold_symm7 = compute_voroni_idx(voro)

    feature_all = np.array(zip(Coordination_number_by_Voroni_tessellation,
                               Coordination_number_by_cutoff_distance,
                               Voroni_idx3, Voroni_idx4, Voroni_idx5, Voroni_idx6, Voroni_idx7,
                               cellfraction,
                               i_fold_symm3, i_fold_symm4, i_fold_symm5, i_fold_symm6, i_fold_symm7,
                               area_weight_i_fold_symm3,
                               area_weight_i_fold_symm4,
                               area_weight_i_fold_symm5,
                               area_weight_i_fold_symm6,
                               area_weight_i_fold_symm7
                               ))

    feature_all_file = 'old_feature-' + tem1 + '.csv'
    pd.DataFrame(feature_all).to_csv('../' + path + '/some old feature SRO/' + feature_all_file)
