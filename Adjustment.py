from FileHandler import *
from DataHandler import *
import sys
from numpy import matmul, transpose
from scipy.linalg import inv
from scipy import stats
from PyQt5 import QtCore

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, suppress=True, precision=2)

NETWORK_DEFECT = 3

# TODO FINISH DOC STRINGS


def run2d(params, communication):
    """
    Main adjustment function.
    :param dict params: Dictonary containing user define parameters from the UI.
    :param QtCore.pyqtSignal communication: MainWindow's msg_signal. Used for communicating with the user.
    :return bool: True when adjustment is finished successfully, false otherwise.
    """
    # [print(k, v) for k, v in params.items()]
    wb = get_workbook(params.get('inputFile'), communication)
    excel_points = extract_excel_data(wb, params.get('PtsDataStart'), params.get('Worksheet'), communication)
    if excel_points == 'Error':
        return False

    points_list = Point.to_point_list(excel_points)
    points_copy = Point.to_point_list(excel_points)  # for future use in Word report without any changes
    points_rename_record = Point.rename_points_list(points_list)
    points_number = len(points_list)

    excel_directions = extract_excel_data(wb, params.get('DirDataStart'), params.get('Worksheet'),
                                          communication, extraction_type='directions')
    if excel_directions == 'Error':
        return False
    directions_list = Direction.to_direction_list(excel_directions)
    Direction.rename_directions_list(directions_list, points_rename_record)

    station_num = get_num_of_stations(directions_list, points_list)
    dir_ni = get_ni_from_coords(points_list, directions_list, communication)
    if dir_ni == 'Error':
        return False
    dir_d = get_distance_from_coords(points_list, directions_list, communication)
    if dir_d == 'Error':
        return False

    for dire in directions_list:
        dire.set_aij()
        dire.set_bij()

    row_length = 2 * points_number
    mat_a_direction_part = make_mat_a_rows(directions_list, row_length,
                                           station_num=station_num)

    excel_distances = extract_excel_data(wb, params.get('DisDataStart'), params.get('Worksheet'),
                                         communication, extraction_type='distances')
    if excel_distances == 'Error':
        return False
    distances_list = Distance.to_distance_list(excel_distances)
    Distance.rename_distance_list(distances_list, points_rename_record)
    dis_ni = get_ni_from_coords(points_list, distances_list, communication)
    if dis_ni == 'Error':
        return False
    dis_d = get_distance_from_coords(points_list, distances_list, communication)
    if dis_d == 'Error':
        return False

    for dis in distances_list:
        dis.set_aij()
        dis.set_bij()

    row_length = 2 * points_number + station_num  # Adds 0 for the columns that represent the Z's
    mat_a_distance_part = make_mat_a_rows(distances_list, row_length)

    a = mat_a(mat_a_direction_part, mat_a_distance_part, communication)
    if type(a) is str:
        return False

    f = fill_mat_f(directions_list, distances_list, points_list)  # also adds value to z0 attr of Point

    p = fill_mat_p(params, len(directions_list), distances_list)

    N = matmul(matmul(transpose(a), p), a)

    n = matmul(matmul(transpose(a), p), f)

    excel_export_data = {}
    global bt_, rt_
    if params.get('datumMethod') == 'min_trace':
        N_ext, bt_, bt_error = bt(N, params['datumCoords'], points_list, NETWORK_DEFECT, points_rename_record)
        excel_export_data.update({'Bt': bt_})
        if bt_error:
            communication.emit('c', 'Greška u generisanju matrice datumskih uslova B!\n'
                                    'Proverite ulazne parametre vezane za tačke i datum mreže.')
            return False
    else:
        N_ext, rt_, rt_error = rt(N, params['datumCoords'], NETWORK_DEFECT, points_rename_record)
        excel_export_data.update({'Rt': rt_})
        if rt_error:
            communication.emit('c', 'Greška u generisanju matrice datumskih uslova R!\n'
                                    'Proverite ulazne parametre vezane za tačke i datum mreže.')

    qx_ext = inv(N_ext)
    qx = remove_datum_extension(qx_ext, NETWORK_DEFECT)

    x = matmul(-qx, n)

    v = matmul(a, x) + f

    control = matrix_control(p, f, n, v, x)
    if not control[0]:
        communication.emit('c', f'Greška u kontroli računanja!\nvTPv: {round(control[1], 4)}\n'
                                f'fTPf + nTx: {round(control[2], 4)}')
        return False

    deg_of_freedom = a.shape[0] - a.shape[1] + NETWORK_DEFECT
    m0 = sqrt((matmul(matmul(transpose(v), p), v)) / deg_of_freedom)
    fisher_distribution = stats.f.isf(float(params['alphaCoef']), deg_of_freedom, 10000)
    t = m0 ** 2 / params['sigma0'] ** 2

    p_inv = inv(p)
    qv = p_inv - matmul(matmul(a, qx), transpose(a))
    qvii = [qv[i][i] if abs(qv[i][i]) > 0.0000000001 else 1 for i in range(qv.shape[0])]
    w = [(i+1, abs(value) / (params['sigma0'] * sqrt(qv))) for i, (value, qv) in enumerate(zip(v, qvii))]

    iter_number = 0
    dropped_measurements = []
    data_snooping_mat = []
    zero_mat = np.zeros((NETWORK_DEFECT, NETWORK_DEFECT))
    norm = stats.norm.ppf(1 - float(params['alphaCoef']) / 2)

    while t > fisher_distribution:
        index, w_value = get_rough_error(w)
        index -= 1  # Decreasing by 1 because it's increased by 1 in w list comp. for aestetic purposes

        if index < len(directions_list):
            dropped_measurements.append(['Pravac', directions_list[index], t, fisher_distribution, w_value, norm])
            directions_list.remove(directions_list[index])
        else:
            dis_index = index - len(directions_list)
            dropped_measurements.append(['Dužina', distances_list[dis_index], t, fisher_distribution, w_value, norm])
            distances_list.remove(distances_list[dis_index])

        data_snooping_mat.append({
            'iteracija': iter_number,
            'A': a,
            'f': f,
            'P': p,
            'N': N,
            'malo n': n,
            'Qx': qx,
            'x': x,
            'v': v,
            'Qv': qv,
            'm0': m0,
            'T': t,
            'F': fisher_distribution,
            'w': np.array(w),
            'Norm': norm,
            'izbaceno merenje': dropped_measurements[-1]
        })

        a, p = remove_rough_error(a, p, index)

        f = fill_mat_f(directions_list, distances_list, points_list)

        N = matmul(matmul(transpose(a), p), a)

        n = matmul(matmul(transpose(a), p), f)

        if params.get('datumMethod') == 'min_trace':
            bt_plus_zero = np.concatenate((bt_, zero_mat), axis=1)
            N_ext = np.concatenate((np.concatenate((N, bt_.transpose()), axis=1), bt_plus_zero), axis=0)
        else:
            rt_plus_zero = np.concatenate((rt_, zero_mat), axis=1)
            N_ext = np.concatenate((np.concatenate((N, rt_.transpose()), axis=1), rt_plus_zero), axis=0)

        qx_ext = inv(N_ext)

        qx = remove_datum_extension(qx_ext, NETWORK_DEFECT)

        x = matmul(-qx, n)

        v = matmul(a, x) + f

        control = matrix_control(p, f, n, v, x)
        if not control[0]:
            communication.emit('c', f'Greška u kontroli računanja!\nvTPv: {round(control[1], 4)}\n'
                                    f'fTPf + nTx: {round(control[2], 4)}')
            return False

        deg_of_freedom = a.shape[0] - a.shape[1] + NETWORK_DEFECT
        m0 = sqrt((matmul(matmul(transpose(v), p), v)) / deg_of_freedom)
        fisher_distribution = stats.f.isf(float(params['alphaCoef']), deg_of_freedom, 10000)
        t = m0 ** 2 / params['sigma0'] ** 2

        p_inv = inv(p)
        qv = p_inv - matmul(matmul(a, qx), transpose(a))
        qvii = [qv[i][i] if abs(qv[i][i]) > 0.0000000001 else 1 for i in range(qv.shape[0])]
        w = [(i + 1, abs(value) / (params['sigma0'] * sqrt(qv))) for i, (value, qv) in enumerate(zip(v, qvii))]

        iter_number += 1

    excel_export_data.update({'data snooping': data_snooping_mat})

    apply_corrections(x, points_list)
    standard_deviations(qx, m0, points_list)

    definitive_control(points_list, v, directions_list, distances_list, communication)

    error_ellipse(qx, m0, points_list, params['alphaCoef'])
    ellipses_scr_times10 = [point.ellipse.to_autocad_scr(point, scale=5) for point in points_list
                            if point.ellipse.b != 0]

    ql = matmul(matmul(a, qx), a.transpose())
    qv = inv(p) - ql

    local_measure(qv.diagonal(), p.diagonal(), directions_list, distances_list)

    marginal_error(qv.diagonal(), p.diagonal(), m0, params['alphaCoef'], params['betaCoef'],
                   directions_list, distances_list)

    excel_export_data.update({
        'A': a,
        'P': p,
        'f': f,
        'N': N,
        'vektor n': n,
        'Qx': qx,
        'x': x,
        'v': v,
        'standardne devijacije': None,
        'elipse gresaka': None,
        'ql': ql,
        'qv': qv,
        'pouzdanost': None
    })

    # Reverting the renaming done in beginning
    Point.revert_rename(points_list, points_rename_record)
    Direction.revert_rename(directions_list, points_rename_record)
    Distance.revert_rename(distances_list, points_rename_record)

    try:
        if word_report(params['wordOutput'], points_copy, points_list, directions_list, distances_list,
                       dropped_measurements, NETWORK_DEFECT, station_num, params, communication,
                       excel_export_data) == 'Error':
            return False
    except NameError:
        if word_report(params['wordOutput'], points_copy, points_list, directions_list, distances_list,
                       dropped_measurements, NETWORK_DEFECT, station_num, params, communication,
                       excel_export_data) == 'Error':
            return False

    if point_csv(points_list, params['csvOutput'], communication) == 'Error':
        return False

    if scr_file(ellipses_scr_times10, params['scrOutput'], communication) == 'Error':
        return False

    if excel_export(excel_export_data, params['excelOutput'], communication,
                    points_list, directions_list, distances_list) == 'Error':
        return False

    return True
