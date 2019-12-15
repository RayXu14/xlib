import math

def lon8lat_to_mercator(lon, lat):

    k = 6378137.
    x = lon * (k * math.pi/180.)
    y = math.log(math.tan((90. + lat) * math.pi/360.)) * k

    return x, y
