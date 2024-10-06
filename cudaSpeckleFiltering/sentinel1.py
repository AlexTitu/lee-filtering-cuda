from esa_snappy import ProductIO
import matplotlib.pyplot as plt
import numpy as np
import math
import glob

fisier = 'subset_0_of_S1B_IW_GRDH_1SDV_20211003T135159_20211003T135224_028972_03751C_2107.dim'

p = ProductIO.readProduct(fisier)
bands = list(p.getBandNames())
imagine = fisier.split('.')[0]
for band_num in range(len(bands)):
    Bi = p.getBand(bands[band_num])
    w = Bi.getRasterWidth()
    h = Bi.getRasterHeight()
    Bi_data = np.zeros(w * h, np.float32)  # 16bits->img 12bits/band
    Bi.readPixels(0, 0, w, h, Bi_data)
    Bi_data.shape = h, w
    np.save("band-{}".format(bands[band_num]), Bi_data)





