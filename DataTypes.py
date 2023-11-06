from math import *
from typing import Tuple, List, Dict


def _get_key(d, value) -> str | int:
    """
    Returns a dictionary key of the first item that matches the given value
    :param dict d: Dictonary with the wanted key
    :param str value: Value of wanted key
    """
    return [k for k, v in d.items() if v == value][0]


def _dms_to_decimal(dms):
    """
    :param List[int, int, float] dms: List or tuple containing the degrees, minutes and seconds of an angle.
    :returns float: Returns decimal value of a given angle. Value is in radians.
    """
    d, m, s = dms
    return radians(d + m/60 + s/3600)


def _decimal_to_dms(angle) -> Tuple[int, int, float]:
    """
    :param float angle:
    :returns: Returns degrees, minutes and seconds of an angle in decimal format.
    """
    try:
        d = int(angle)
        m = (angle - d) * 60
        s = (m - int(m)) * 60
        return d, int(m), round(s, 1)
    except ValueError:
        return 0, 0, 0


class ErrorEllipse:
    def __init__(self, theta, a, b):
        self.theta = theta
        self.a = a
        self.b = b
        self.thetaDMS = _decimal_to_dms(self.theta)

    def __str__(self):
        return f'\u03B8: {self.theta} ; A: {self.a} ; B: {self.b}\n'

    def __repr__(self):
        return f'\u03B8: {self.theta} ; A: {self.a} ; B: {self.b}\n'

    def to_autocad_scr(self, point, scale=1):
        """
        :param Point point: Instance of Point class
        :param float scale: Multiplier for ellipse size for clearer visualization in CAD
        :returns str: Returns a string in the format of AutoCad script file (*.scr) for ellipse drawing for given point.
        """
        return f'ELLIPSE\n' \
               f'C\n' \
               f'{point.y},{point.x}\n' \
               f'{point.y + self.a*scale * sin(radians(self.theta))},' \
               f'{point.x + self.a*scale * cos(radians(self.theta))}'\
               f'\n{self.b*scale}'


class Point(object):
    def __init__(self, point_data) -> None:
        point_id, y, x = point_data
        self.id = str(point_id)
        self.y = y
        self.x = x
        self.is_station = False
        self.z0 = None
        self.sigma_y = None
        self.sigma_x = None
        self.sigma_p = None
        self.ellipse = None

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(f"Greška prilikom računanja sa tačkom '{self.id}'. Proverite početne ćelije.")

    def __str__(self) -> str:
        return f'\nID: {self.id}, Y = {self.y}, X = {self.x}'
    
    def __repr__(self) -> str:
        return f'\nID: {self.id}, Y = {self.y}, X = {self.x}'

    def set_id(self, name: str | int):
        """
        Sets new point ID
        :param name:
        :return None:
        """
        self.id = name

    def set_y(self, y):
        """
        Sets Y value of a point
        :param float y:
        :return None:
        """
        self.y = y

    def set_x(self, x):
        """
        Sets X value of a point
        :param x:
        :return None:
        """
        self.x = x

    def set_is_station(self):
        """
        Set is_station flag of a point to True.
        :return None:
        """
        self.is_station = True

    def set_z0(self, z0):
        """
        Sets Z0 (Zenith angle) value of a point
        :param float z0: Zenith angle value
        :return None:
        """
        self.z0 = z0

    def set_sigma_y(self, sigma_y):
        """
        Sets sigma Y value of a Point
        :param float sigma_y: Standard deviation of Y coordinate
        :return None:
        """
        self.sigma_y = sigma_y

    def set_sigma_x(self, sigma_x):
        """
        Sets sigma X value of a Point
        :param float sigma_x: Standard deviation of X coordinate
        :return None:
        """
        self.sigma_x = sigma_x

    def set_sigma_p(self):
        """
        Calculates and sets sigma P value of a Point. Sigma P - positional standard deviation sqrt(sigmaY^2 + sigmaX^2)
        :return None:
        """
        self.sigma_p = sqrt(self.sigma_y**2 + self.sigma_x**2)

    def set_ellipse(self, ellipse):
        """
        Sets the ellipse value of a Point
        :param ErrorEllipse ellipse: instance of ErrorEllipse class
        :return None:
        """
        self.ellipse = ellipse

    def distance(self, point) -> Tuple[str, float]:
        """
        Calculates the Euclidean distance between the point it's called on and the passed point.
        :param Point point: instance of Point class
        :returns: Tuple in the format ('from-to', value). From is the ID of point the method is called
        on, to is the ID of passed point.
        :raises ValueError: If no point is provided or the provided parameter is not of Point type
        """
        if point is None or type(point) is not Point:
            raise ValueError('Tačka ne postoji. Proverite početne ćelije')

        return f'{self.id}-{point.id}', sqrt((point.y - self.y) ** 2 + (point.x - self.x) ** 2)

    def ni(self, point) -> Tuple[str, float]:
        """
        Calculates the directional angle between the point it's called on and the passed point.
        :param point: Instance of Point class
        :returns: Tuple in the format ('from-to', value). From is the ID of point the method is called
        on, to is the ID of passed point.
        :raises ValueError: If no point is provided or the provided parameter is not of Point type.
        :raises TypeError: If during the coordinates differences calculation types aren't subtractable.
        """
        if point is None or type(point) is not Point:
            raise ValueError('Tačka ne postoji. Proverite početne ćelije')
        try:
            dy = point.y - self.y
            dx = point.x - self.x
        except TypeError:
            raise TypeError('Nemoguće računanje razlika koordinata.\nProverite vrednosti ćelija koordinata tačaka'
                            '\nili vrednost početne ćelije tačaka.')

        try:
            if dy >= 0 and dx >= 0:
                return f'{self.id}-{point.id}', atan(abs(dy/dx))
            elif dy > 0 > dx:
                return f'{self.id}-{point.id}', atan(abs(dx/dy)) + pi/2
            elif dy < 0 and dx < 0:
                return f'{self.id}-{point.id}', atan(abs(dy/dx)) + pi
            else:
                return f'{self.id}-{point.id}', atan(abs(dx/dy)) + 3*pi/2
        except ZeroDivisionError:
            dy = self.y - point.y
            dx = self.x - point.x

            if dy >= 0 and dx >= 0:
                ni = atan(abs(dy/dx))
            elif dy > 0 > dx:
                ni = atan(abs(dx/dy)) + pi/2
            elif dy < 0 and dx < 0:
                ni = atan(abs(dy/dx)) + pi
            else:
                ni = atan(abs(dx/dy)) + 3*pi/2

            ni += pi
            if ni > 2*pi:
                ni -= 2*pi
            return f'{self.id}-{point.id}', ni

    def toCSV(self) -> str:
        """
        :returns: Point ID, Y, X coordinate as a string delimited by a comma.
        """
        return f'{self.id},{self.y},{self.x}\n'

    @staticmethod
    def list_to_dict(pts_list) -> dict:
        """
        Static method used to convert a Point list to a Point dictonary. Keys are integers starting from 1.
        :param List[Point] pts_list:
        :returns: Dictonary containing instances if Point class provided in the list as dictonary values.
        """
        return {i+1: point for i, point in enumerate(pts_list)}

    @staticmethod
    def to_point_list(excel_data):
        """
        Static method used for converting data extracted from an Excel file into a list of instances of Point class
        :param List[List[str, float, float]] excel_data: nested list in which each inner list represent a point params.
        :return: List of Point created from input data
        """
        return [Point(data) for data in excel_data]

    @staticmethod
    def get_point_from_id(points_list, point_id):
        """
        Static method used for finding a point in a list based on its ID.
        :param List[Point] points_list: list of instances of Point class from which then point is desired from.
        :param int point_id: ID of desired point.
        :return Point: Point that matches the ID provided
        """
        return [point for point in points_list if point.id == point_id][0]

    @staticmethod
    def rename_points_list(point_list):
        """
        Static method used for changing the ID's of points so that the user is not limited to use integer numbers for
        point ID's. It assigns integer numbers, starting from 1, to each point and creates a renaming record.
        :param List[Point] point_list: List of instances of Point class
        :return: Dictonary (rename record) with key being the original point ID and the value being the new point ID.
        """
        new_names_record = {}
        for new_name, point in enumerate(point_list):
            new_name += 1  # 1, 2, 3, ...
            original_name = point.id
            point.set_id(new_name)
            new_names_record.update({original_name: new_name})
        return new_names_record
    
    @staticmethod
    def revert_rename(point_list, rename_record):
        """
        Static method used to reverts the changes done to the ID's of points based on the provided rename record from
        the rename_points_list() method.
        :param List[Point] point_list:
        :param Dict[str, int] rename_record:
        :return None:
        """
        if len(point_list) != len(rename_record):
            print('check if Point list matches Point name record')
            return
        
        for point in point_list:
            if point.id in rename_record.values():
                point.set_id(_get_key(rename_record, point.id))
            else:
                print('check if Point list matches Point name record')


class Distance(object):
    def __init__(self, distance_data) -> None:
        from_, to, value = distance_data
        self.from_ = from_
        self.to = to
        self.value = value

    def __str__(self) -> str:
        return f'\n{self.from_}-{self.to} : {self.value}'
    
    def __repr__(self) -> str:
        return f'\n{self.from_}-{self.to} : {self.value}'
    
    def set_from(self, name):
        self.from_ = name

    def set_to(self, name):
        self.to = name

    @staticmethod
    def rename_distance_list(distance_list, points_rename_record: dict):
        for d in distance_list:
            if str(d.from_) in points_rename_record.keys():
                d.set_from(points_rename_record.get(str(d.from_)))
            if str(d.to) in points_rename_record.keys():
                d.set_to(points_rename_record.get(str(d.to)))

    @staticmethod
    def revert_rename(distance_list, points_rename_record: dict):
        for d in distance_list:
            if d.from_ in points_rename_record.values():
                d.set_from(_get_key(points_rename_record, d.from_))
            if d.to in points_rename_record.values():
                d.set_to(_get_key(points_rename_record, d.to))
        return

    @staticmethod
    def to_distance_list(data_list):
        return [Distance(data) for data in data_list]


class Direction(object):
    def __init__(self, direction_data) -> None:
        from_, to, degrees, minutes, seconds = direction_data
        self.from_ = from_
        self.to = to
        self.valueDMS = [degrees, minutes, seconds]
        self.value = _dms_to_decimal(self.valueDMS)

    def __str__(self) -> str:
        return f'\n{self.from_}-{self.to} : {self.valueDMS[0]}° {self.valueDMS[1]}\' {self.valueDMS[2]}"'
    
    def __repr__(self) -> str:
        return f'\n{self.from_}-{self.to} : {self.valueDMS[0]}° {self.valueDMS[1]}\' {self.valueDMS[2]}"'
        
    def set_from(self, name):
        self.from_ = name

    def set_to(self, name):
        self.to = name

    @staticmethod
    def rename_directions_list(direction_list, points_rename_record: dict):
        for d in direction_list:
            if str(d.from_) in points_rename_record.keys():
                d.set_from(points_rename_record.get(str(d.from_)))
            if str(d.to) in points_rename_record.keys():
                d.set_to(points_rename_record.get(str(d.to)))

    @staticmethod
    def revert_rename(direction_list, points_rename_record: dict):
        for d in direction_list:
            if d.from_ in points_rename_record.values():
                d.set_from(_get_key(points_rename_record, d.from_))
            if d.to in points_rename_record.values():
                d.set_to(_get_key(points_rename_record, d.to))
        return

    @staticmethod
    def to_direction_list(data_list):
        return [Direction(data) for data in data_list]
