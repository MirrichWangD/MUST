import math
import numpy as np


a = 6378137.0
f = 1 / 298.257223565


# latitude:纬度 longitude:经度 altitude:海拔
def LLA_to_XYZ(latitude, longitude, altitude):
    # 经纬度的余弦值
    cosLat = math.cos(latitude * math.pi / 180)
    sinLat = math.sin(latitude * math.pi / 180)
    cosLon = math.cos(longitude * math.pi / 180)
    sinLon = math.sin(longitude * math.pi / 180)

    # WGS84坐标系的参数
    rad = 6378137.0  # 地球赤道平均半径（椭球长半轴：a）
    f = 1.0 / 298.257224  # WGS84椭球扁率 :f = (a-b)/a
    C = 1.0 / math.sqrt(cosLat * cosLat + (1 - f) * (1 - f) * sinLat * sinLat)
    S = (1 - f) * (1 - f) * C
    h = altitude

    # 计算XYZ坐标
    X = (rad * C + h) * cosLat * cosLon
    Y = (rad * C + h) * cosLat * sinLon
    Z = (rad * S + h) * sinLat

    N = a / np.sqrt(1 - f * (2 - f) * sinLat**2)

    X = (N + altitude) * cosLat * cosLon
    Y = (N + altitude) * cosLat * sinLon
    Z = (N * (1 - f) ** 2 + altitude) * sinLat

    return np.array([X, Y, Z])


def XYZ_to_LLA(X, Y, Z):
    # WGS84坐标系的参数
    a = 6378137.0  # 椭球长半轴
    b = 6356752.314245  # 椭球短半轴
    ea = np.sqrt((a**2 - b**2) / a**2)
    eb = np.sqrt((a**2 - b**2) / b**2)
    p = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Z * a, p * b)

    # 计算经纬度及海拔
    longitude = np.arctan2(Y, X)
    latitude = np.arctan2(Z + eb**2 * b * np.sin(theta) ** 3, p - ea**2 * a * np.cos(theta) ** 3)
    N = a / np.sqrt(1 - ea**2 * np.sin(latitude) ** 2)
    altitude = p / np.cos(latitude) - N

    return np.array([np.degrees(latitude), np.degrees(longitude), altitude])


# 根据经度、纬度、海拔计算两地距离:法一
def get_distance1(position1, position2):
    v1 = LLA_to_XYZ(position1[0], position1[1], position1[2])
    v2 = LLA_to_XYZ(position2[0], position2[1], position2[2])
    distance = np.linalg.norm(v1 - v2)

    return distance


# 根据经度、纬度计算两地距离：法二
def get_distance2(lat1, lon1, lat2, lon2):
    """获取地理坐标系下的两点间距离"""
    # GetDistanceInGeographyCoordinate, return two point distance
    radius_lat1 = lat1 * math.pi / 180
    radius_lat2 = lat2 * math.pi / 180
    radius_lon1 = lon1 * math.pi / 180
    radius_lon2 = lon2 * math.pi / 180
    a = radius_lat1 - radius_lat2
    b = radius_lon1 - radius_lon2
    distance = 2 * math.asin(
        math.sqrt(pow(math.sin(a / 2.0), 2) + math.cos(radius_lat1) * math.cos(radius_lat2) * pow(math.sin(b / 2.0), 2))
    )

    distance = distance * 6378137
    distance = distance - (distance * 0.0011194)
    return distance


def get_bearing(lat1, lon1, lat2, lon2):
    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360

    return bearing


latitude = 22.138214
longitude = 113.538533
altitude = 37.271128


x, y, z = LLA_to_XYZ(latitude, longitude, altitude)
print(x, y, z)

lat, lon, alt = XYZ_to_LLA(x, y, z)
print(lat, lon, alt)
