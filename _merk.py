from geo_helper import num_with_relatives2mercator_3395, merkator2num_with_relatives_3395, num_with_relatives2deg_3395, num2deg, deg2merkator_3395

tile_x = 76538
tile_y = 38197
rx = 123
ry = 32
zoom = 17


lon, lat = num_with_relatives2deg_3395(tile_x, tile_y, rx, ry, zoom)
print(lon, lat)

mx, my = deg2merkator_3395(lon, lat)
tx, ty, x, y = merkator2num_with_relatives_3395(mx, my, zoom)
print(tx, ty, x, y)


llon, llat = num2deg(tile_x, tile_y, zoom)
print(llon, llat)

print('!!!!')
