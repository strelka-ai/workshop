from geo_helper import num_and_relatives2mercator_3395, merkator2num_with_relatives_3395


mx, my = num_and_relatives2mercator_3395(76538, 38197, 123, 32, 17)
tx, ty, x, y = merkator2num_with_relatives_3395(mx, my, 17)

print(tx, ty, x, y)
