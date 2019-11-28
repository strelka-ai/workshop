import math

RADIUS3395 = 6378137.0  # ellipsoid radius on equator
EXTRNC3395 = 0.0818191908426  # eccentricity
EQUATOR3395 = 40075016.685578488  # EPSG:3395 equator length


def num2deg(tile_x, tile_y, zoom):
    '''

    Convert tile numners into degrees of longitude and latitude

    :param tile_x:
    :param tile_y:
    :param zoom:
    :return:
    '''
    n = 2.0 ** zoom
    lon_deg = tile_x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
    lat_deg = math.degrees(lat_rad)

    return lon_deg, lat_deg


def merkator2num_3395(merkator_x, merkator_y, zoom):
    '''

    Convert merkator into the tile number x and y 3395 std.

    :param merkator_x:
    :param merkator_y:
    :param zoom:
    :return:
    '''
    tile_x, tile_y, _, _ = merkator2num_with_relatives_3395(merkator_x, merkator_y, zoom)
    return tile_x, tile_y


def merkator2num_with_relatives_3395(merkator_x, merkator_y, zoom, relative_tile_size=256):
    '''

    Convert merkator into the tile x and y numbers 3395 std. with additional relative coordinates of the point

    :param merkator_x:
    :param merkator_y:
    :param zoom:
    :param relative_tile_size: the size of the tile (assumed that the tile will be square)
    :return:
    '''

    total_tiles = math.pow(2, zoom)
    tile_size = total_tiles / EQUATOR3395

    tile_fx = (EQUATOR3395 / 2.0 + merkator_x) * tile_size
    tile_fy = (EQUATOR3395 / 2.0 - merkator_y) * tile_size

    tile_x = int(math.floor(tile_fx))
    tile_y = int(math.floor(tile_fy))

    relative_x = int(math.floor((tile_fx - tile_x) * relative_tile_size))
    relative_y = int(math.floor((tile_fy - tile_y) * relative_tile_size))

    return tile_x, tile_y, relative_x, relative_y


def num_and_relatives2mercator_3395(tile_x, tile_y, relative_x, relative_y, zoom, relative_tile_size=256):
    """

    Convert tile and relative coords to merkator 3395

    :param tile_x:
    :param tile_y:
    :param relative_x:
    :param relative_y:
    :param zoom:
    :return:
    """

    total_tiles = math.pow(2, zoom)
    tile_size = total_tiles / EQUATOR3395

    tile_fx = tile_x + relative_x / relative_tile_size
    tile_fy = tile_y + relative_y / relative_tile_size

    merkator_x = tile_fx / tile_size - EQUATOR3395 / 2.0
    merkator_y = EQUATOR3395 / 2.0 - tile_fy / tile_size

    return merkator_x, merkator_y


def merkator2deg_3395(merkator_x, merkator_y):
    '''

    Convert merkator into degrees using 3395 std (for Yandex Maps)

    :param merkator_x:
    :param merkator_y:
    :return:
    '''

    # latlong in radians
    lat_rad = radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # reproject latlong to EPSG:3395
    esin_lat = EXTRNC3395 * math.sin(lat_rad)
    tan_temp = math.tan(math.pi / 4 + lat_rad / 2)
    pow_temp = math.pow(math.tan(math.pi / 4 + math.asin(esin_lat) / 2), EXTRNC3395)
    U = tan_temp / pow_temp

    merkator_lat = RADIUS3395 * math.log(U)
    merkator_lon = (RADIUS3395 * lon_rad)

    return merkator_lon, merkator_lat


def deg2merkator_3395(lon_deg, lat_deg):
    '''

    Convert degrees into merkator using 3395 std (for Yandex Maps)

    :param lon_deg:
    :param lat_deg:
    :return:
    '''

    # latlong in radians
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # reproject latlong to EPSG:3395
    esin_lat = EXTRNC3395 * math.sin(lat_rad)
    tan_temp = math.tan(math.pi / 4 + lat_rad / 2)
    pow_temp = math.pow(math.tan(math.pi / 4 + math.asin(esin_lat) / 2), EXTRNC3395)
    U = tan_temp / pow_temp

    merkator_lat = RADIUS3395 * math.log(U)
    merkator_lon = (RADIUS3395 * lon_rad)

    return merkator_lon, merkator_lat


def deg2num_3395(lon_deg, lat_deg, zoom):
    '''

    Convert degrees into the tile x and y numbers for Yandex Maps

    :param lon_deg: longitude degree
    :param lat_deg: lattitude degree
    :param zoom: zoom value
    :return: tile_x:, tile_y:
    '''
    merkator_lon, merkator_lat = deg2merkator_3395(
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        # zoom=zoom
    )

    return merkator2num_3395(
        merkator_x=merkator_lon,
        merkator_y=merkator_lat,
        zoom=zoom
    )
