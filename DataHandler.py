from typing import List, Tuple
from math import *
import numpy as np
from numpy import ndarray
from DataTypes import Point, Direction, Distance, ErrorEllipse
from scipy.stats import chi2, t

RO = (180 / pi) * 3600
_360_DEG_IN_SEC = 1296000
DATUM_EXT_ERROR_TOLERANCE_MAT = np.array([[0.000001, 0.000001, 0.000001],
                                          [0.000001, 0.000001, 0.000001],
                                          [0.000001, 0.000001, 0.000001]]).any()


def _avg(numbers):
    return sum(numbers)/len(numbers)


def _quadrant_atan(a, b):
    if a == b == 0:
        return 0

    if a >= 0 and b >= 0:
        return atan(a / b)
    elif a > 0 > b:
        return atan(a / b) + pi
    elif a < 0 and b < 0:
        return atan(a / b) + pi
    else:
        return atan(a / b) + 2 * pi


def get_ni_from_coords(pts_list, msm_list, comm_signal, adjusted=False) -> None | str:
    """
    Updates Directions/Distances ni_from_coords variable.
    :param comm_signal: pyqtSignal used for communication though the program
    :param List[Point] pts_list: List of Points
    :param List[Distance] | List[Direction] msm_list: List of Directions/Distances
    :param bool adjusted: True if updates adj_ni_from_coords variable (Adjusted points are used), False by default
    :returns:  None/'Error'
    """
    # This only works because list rename methods because they use 1 2 3... as Point ID
    if len(pts_list) == 0:
        comm_signal.emit('c', 'Ne postoje podaci o tačkama. Proverite početnu ćeliju tačaka.')
        return 'Error'

    try:
        for measurement in msm_list:
            station = Point.get_point_from_id(pts_list, measurement.from_)
            to = Point.get_point_from_id(pts_list, measurement.to)

            if adjusted:
                measurement.set_adj_ni_from_coords(station.ni(to))
            else:
                measurement.set_ni_from_coords(station.ni(to))

    except AttributeError:
        comm_signal.emit('c', f'Greška prilikom računanja direkcionog ugla.\n'
                              f'Proverite početne ćelije i ime radnog lista sa podacima.')
        return 'Error'
    except ValueError as e:
        comm_signal.emit('c', e.args[0])
        return 'Error'
    except TypeError as e:
        comm_signal.emit('c', e.args[0])
        return 'Error'


def get_distance_from_coords(pts_list, msm_list, comm_signal, adjusted=False) -> None | str:
    """
    Updates Directions/Distances d_from_coords variable.

    :param comm_signal: pyqtSignal used for communication though the program
    :param List[Point] pts_list: List of Points
    :param List[Distance] | List[Direction] msm_list: List of Directions/Distances
    :param bool adjusted: True if updates adj_d_from_coords variable (Adjusted points are used), False by default
    :returns:  None/'Error'
    """
    # This only works because list rename methods because they use 1 2 3... as Point ID
    # pts_dict = {i + 1: point for i, point in enumerate(pts_list)}
    try:
        for measurement in msm_list:
            station = Point.get_point_from_id(pts_list, measurement.from_)
            to = Point.get_point_from_id(pts_list, measurement.to)

            if adjusted:
                measurement.set_adj_d_from_coords(station.distance(to))
            else:
                measurement.set_d_from_coords(station.distance(to))

        # return [pts_dict.get(measurement.from_).distance(pts_dict.get(measurement.to)) for measurement in msm_list]
    except AttributeError:
        comm_signal.emit('c', f'Greška prilikom računanja rastojanja.\n'
                              f'Proverite početne ćelije i ime radnog lista sa podacima.')
        return 'Error'
    except ValueError as e:
        comm_signal.emit('c', e.args[0])


def get_num_of_stations(directions: List[Direction], points: List[Point]) -> int:
    """
    Calculates the number of stations based on directions measured.
    Changes the 'is_station' attribute of the Point to True.
    :return: Number of stations as an Integer value
    """

    station_list = np.unique(np.array([d.from_ for d in directions]))
    [point.set_is_station() for point in points if point.id in station_list]

    return len(station_list)


def make_mat_a_rows(msm_list: List[Direction] | List[Distance], row_len, station_num=0):
    ab_part = []
    z_part = []
    for msm in msm_list:
        row = [0] * row_len
        # MUST DO RENAME TO WORK

        y_index = 2 * msm.from_ - 2
        x_index = y_index + 1
        row[y_index] = msm.bij
        row[x_index] = msm.aij

        y_index = 2 * msm.to - 2
        x_index = y_index + 1
        row[y_index] = -msm.bij
        row[x_index] = -msm.aij

        ab_part.append(row)

        if station_num != 0:
            z_row = [0] * station_num
            z_row[msm.from_ - 1] = 1
            z_part.append(z_row)

    if station_num != 0:
        return np.concatenate((np.array(ab_part), z_part), axis=1)
    else:
        return np.array(ab_part)


def mat_a(direction_coef, distance_coef, comm_signal):
    try:
        return np.concatenate((direction_coef, distance_coef), axis=0)
    except ValueError:
        comm_signal.emit('c', f'Greška prilikom spajanja dela matrice \'A\' sa koef. pravaca \n'
                              f'i dela sa koef. dužina.\n'
                              f'Proverite početne ćelije i ime radnog lista sa podacima.')
        return 'Error'


def fill_mat_f(directions: List[Direction], distances: List[Distance], points):
    f_dir = []

    for dire in directions:
        if (dire.value - dire.ni_from_coords) <= 0:
            dire.set_z(dire.value - dire.ni_from_coords + 2 * pi)
        else:
            dire.set_z(dire.value - dire.ni_from_coords)

    z_prev = 0
    same_station = []
    for i, dire in enumerate(directions):
        if i == 0:
            z_prev = dire.from_

        if z_prev == dire.from_:
            same_station.append(dire.z)
        else:
            Point.get_point_from_id(points, int(z_prev)).set_z0(_avg(same_station))
            same_station.clear()
            same_station.append(dire.z)

        if i == (len(directions) - 1):
            Point.get_point_from_id(points, int(z_prev)).set_z0(_avg(same_station))
        z_prev = dire.from_

    for dire in directions:
        alpha = degrees((dire.ni_from_coords + Point.get_point_from_id(points, dire.from_).z0)
                        - dire.value - 2 * pi) * 3600

        if alpha > 1000000:
            alpha -= _360_DEG_IN_SEC
        elif alpha < -1000000:
            alpha += _360_DEG_IN_SEC

        f_dir.append(alpha)

    f_dis = [(dis.d_from_coords - dis.value) * 1000 for dis in distances]

    [f_dir.append(d) for d in f_dis]

    return np.array(f_dir)


def fill_mat_p(params: dict, dir_num: int, distance_list: List[Distance]) -> ndarray:

    p_dir = [params['sigma0'] ** 2 / (params['sigmaP'] / sqrt(params['DirGyrus'])) ** 2] * dir_num
    p_dis = [params['sigma0'] ** 2 / ((params['sigmaD'][0] + params['sigmaD'][1] * (d.d_from_coords / 1000)) /
                                      sqrt(params['DisRepeat'])) ** 2 for d in distance_list]
    for p in p_dis:
        p_dir.append(p)
    p = np.diag(p_dir)

    p = np.triu(p)
    p = np.tril(p)

    return p


def bt(n: ndarray, datum_coords: List[str], pts_list: List[Point], defect: int, point_rename_record):
    n_len = n.shape[0]
    bt = np.zeros((defect, n_len))

    if 'Sve tačke' in datum_coords:  # when whole matrix gets filled
        pt_len = len(pts_list)
        for i in range(2 * pt_len):
            if i % 2 == 0:
                bt[0][i] = 1 / sqrt(pt_len)
            else:
                bt[1][i] = 1 / sqrt(pt_len)

        y_sum = 0
        x_sum = 0
        for point in pts_list:
            y_sum += point.y
            x_sum += point.x
        y0 = y_sum / pt_len
        x0 = x_sum / pt_len

        each_point_g = []
        for point in pts_list:
            each_point_g.append([point.y - y0, point.x - x0])

        each_point_g2 = []
        for point in each_point_g:
            each_point_g2.append([point[0] ** 2, point[1] ** 2])

        y_g_sum = 0
        x_g_sum = 0
        for point in each_point_g2:
            y_g_sum += point[0]
            x_g_sum += point[1]

        g = sqrt(y_g_sum + x_g_sum)

        eta = [point[0] / g for point in each_point_g]
        xi = [point[1] / g for point in each_point_g]

        for i in range(2 * pt_len):
            if i % 2 == 0:
                index = i
                bt[2][i] = -xi[int(index / 2)]
            else:
                index = i
                bt[2][i] = eta[int(index / 2)]

        bt_test = np.matmul(bt, bt.transpose())
        if (abs(bt_test - np.diag([1, 1, 1])) < DATUM_EXT_ERROR_TOLERANCE_MAT).all():
            bt_error = False
        else:
            bt_error = True

        zero_mat = np.zeros((defect, defect))
        bt_plus_zero = np.concatenate((bt, zero_mat), axis=1)

        return np.concatenate((np.concatenate((n, bt.transpose()), axis=1), bt_plus_zero), axis=0), bt, bt_error
    else:  # when there are 0 in Y and X column of the points which aren't in chosen points list/datum list

        chosen_points_id = [point_rename_record[f'{pt_id}'] for pt_id in datum_coords]
        chosen_points = [Point.get_point_from_id(pts_list, pt_id) for pt_id in chosen_points_id]
        y_sum = 0
        x_sum = 0
        for point in chosen_points:
            y_sum += point.y
            x_sum += point.x
        x0 = x_sum / len(chosen_points)
        y0 = y_sum / len(chosen_points)

        pt_len = len(chosen_points)

        each_point_g = []
        for point in chosen_points:
            each_point_g.append([point.y - y0, point.x - x0])

        each_point_g2 = []
        for point in each_point_g:
            each_point_g2.append([point[0] ** 2, point[1] ** 2])

        y_g_sum = 0
        x_g_sum = 0
        for point in each_point_g2:
            y_g_sum += point[0]
            x_g_sum += point[1]

        g = sqrt(y_g_sum + x_g_sum)

        eta = [point[0] / g for point in each_point_g]
        xi = [point[1] / g for point in each_point_g]

        for i, point in enumerate(chosen_points):
            y_id = 2 * point.id - 2
            x_id = y_id + 1

            bt[0][y_id] = 1 / sqrt(pt_len)
            bt[1][x_id] = 1 / sqrt(pt_len)
            bt[2][y_id] = -xi[i]
            bt[2][x_id] = eta[i]

        bt_test = np.matmul(bt, bt.transpose())
        if (abs(bt_test - np.diag([1, 1, 1])) < DATUM_EXT_ERROR_TOLERANCE_MAT).all():
            bt_error = False
        else:
            bt_error = True

        zero_mat = np.zeros((defect, defect))
        bt_plus_zero = np.concatenate((bt, zero_mat), axis=1)

        return np.concatenate((np.concatenate((n, bt.transpose()), axis=1), bt_plus_zero), axis=0), bt, bt_error


def rt(n: ndarray, datum_coords: str, defect: int, points_rename_record: dict):
    rt = np.zeros((defect, n.shape[0]))
    point, coord = datum_coords.split(';')
    point = points_rename_record[point]

    rt[0][2*point - 2] = 1
    rt[1][2*point - 1] = 1

    coord_letter, coord_num = coord.split('-')
    coord_num = points_rename_record[coord_num]
    minus = 2
    if coord_letter == 'X':
        minus = 1
    rt[2][2*coord_num - minus] = 1

    rt_test = np.matmul(rt, rt.transpose())

    if (abs(rt_test - np.diag([1, 1, 1])) < DATUM_EXT_ERROR_TOLERANCE_MAT).all():
        rt_error = False
    else:
        rt_error = True

    zero_mat = np.zeros((defect, defect))
    rt_plus_zero = np.concatenate((rt, zero_mat), axis=1)
    return np.concatenate((np.concatenate((n, rt.transpose()), axis=1), rt_plus_zero), axis=0), rt, rt_error


def remove_datum_extension(mat: ndarray, defect):
    del_indexes = [mat.shape[0] - i - 1 for i in range(defect)]
    mat = np.delete(mat, del_indexes, 0)
    mat = np.delete(mat, del_indexes, 1)
    return mat


def matrix_control(p, f, n, v, x) -> Tuple[bool, float, float]:
    c1 = np.matmul(np.matmul(v.T, p), v)
    c2 = np.matmul(np.matmul(f.T, p), f) + np.matmul(n.T, x)
    if abs(c1 - c2) > 0.001:
        return False, c1, c2
    else:
        return True, c1, c2


def get_rough_error(w):
    only_values_abs = [i[1] for i in w]
    rough_error = max(only_values_abs)
    for value in w:
        if rough_error == value[1]:
            return value


def remove_rough_error(a: ndarray, p: ndarray, index: int):
    a_new = np.delete(a, index, 0)
    p_new = np.delete(p, index, 0)
    p_new = np.delete(p_new, index, 1)
    return a_new, p_new


def apply_corrections(x, points):
    z0_corrections = x[len(points) * 2:]
    counter = 0
    for point in points:
        point.set_y(point.y + x[point.id * 2 - 2]/1000)
        point.set_x(point.x + x[point.id * 2 - 1]/1000)
        if point.is_station:
            point.set_z0(point.z0 + radians(z0_corrections[counter]/3600))
            counter += 1


def standard_deviations(qx, m0, points):
    qx_diag = np.diag(qx)
    point_counter = 0
    counter = 0
    for i, diag_el in enumerate(qx_diag):
        point = points[point_counter]
        counter += 1
        if i % 2 == 0:
            if diag_el != 0:
                point.set_sigma_y(m0 * sqrt(diag_el))
            else:
                point.set_sigma_y(0)
        else:
            if diag_el != 0:
                point.set_sigma_x(m0 * sqrt(diag_el))
            else:
                point.set_sigma_x(0)

        if counter == 2:
            point.set_sigma_p()
            point_counter += 1
            counter = 0

        if i == (2 * len(points) - 1):
            break


def error_ellipse(qx, m0, points: List, alpha):
    qyy = []
    qxx = []
    qxy = []
    for i, q in enumerate(np.diag(qx)):
        if abs(q) < 0.0000001:
            q = 0

        if i % 2 == 0:

            qyy.append(q)
            if abs(qx[i + 1][i]) < 0.0000001:
                qxy.append(0)
            else:
                qxy.append(qx[i + 1][i])
        else:
            qxx.append(q)

        if i == (2 * len(points) - 1):
            break

    qxy2 = [2 * q for q in qxy]
    qxx_minus_qyy = [q1 - q2 for q1, q2 in zip(qxx, qyy)]

    theta = [degrees(_quadrant_atan(a, b) / 2) for a, b in zip(qxy2, qxx_minus_qyy)]

    k = [sqrt(x_m_y**2 + 4*q**2) for x_m_y, q in zip(qxx_minus_qyy, qxy)]

    lambda1 = []
    lambda2 = []
    [[lambda1.append((y + x + k_) / 2), lambda2.append((y + x - k_) / 2)] for y, x, k_ in zip(qyy, qxx, k)]
    statistics_constant = chi2.ppf(1 - (float(alpha)), 2)

    a = []
    b = []
    for l1, l2 in zip(lambda1, lambda2):
        try:
            a.append(m0 * sqrt(statistics_constant * l1))
        except ValueError:  # Math domain error in cases when l1 is a really small number (10-17) but it's negative
            a.append(0)

        try:
            b.append(m0 * sqrt(statistics_constant * l2))
        except ValueError:
            b.append(0)

    [point.set_ellipse(ErrorEllipse(theta_, a_, b_)) for point, theta_, a_, b_ in zip(points, theta, a, b)]


def local_measure(qv_diag, p_diag, directions, distances):
    for qv, p, msm in zip(qv_diag, p_diag, (directions + distances)):
        msm.set_ri(qv * p)


def marginal_error(qv_diag, p_diag, m0, alpha, beta, directions, distances):
    non_centrality_param = t.ppf(float(beta), 10000) + t.ppf(1 - float(alpha) / 2, 10000)
    for qv, p, msm in zip(qv_diag, p_diag, (directions + distances)):
        if qv < 0:
            qv = abs(qv)
        g = (m0 * non_centrality_param) / (p * sqrt(qv))
        if g > 1000000:
            g = np.inf
        msm.set_gi(g)


def definitive_control(points, v, directions: List[Direction], distances: List[Distance], communication):
    get_ni_from_coords(points, directions, communication, adjusted=True)
    get_distance_from_coords(points, distances, communication, adjusted=True)

    v_dir = v[:len(directions)]
    v_dis = v[len(directions):]
    for dire, vi in zip(directions, v_dir):
        station_point = Point.get_point_from_id(points, dire.from_)
        dir_from_adjusted = dire.adj_ni_from_coords + station_point.z0 - 2*pi

        u_dir = dir_from_adjusted - dire.value

        if u_dir < - 2*pi:
            u_dir += 2*pi

        dire.set_definitive_control(degrees(u_dir * 3600) - vi)

    for dis, vi in zip(distances, v_dis):
        dis.set_definitive_control((dis.adj_d_from_coords - dis.value)*1000 - vi)


