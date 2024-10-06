import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from esa_snappy import ProductIO


def lee_sigma_filter(vec, size, Cu, Cmax):
    """
    Aplica filtrul Lee-Sigma pentru o fereastra locala de pixeli.

    Parameters:
    vec : 2D numpy array
        Vecinatatea de pixeli din jurul pixelului de interes
    size : int
        Dimensiunea ferestrei
    Cu : float
        Constanta universala pentru scalarea zgomotului, de obicei in jur de 0.523.
    Cmax : float
        Coeficientul maxim de varianta a zgomotului, definirea pragului de similaritate pentru varianta.

    Returns:
    float
        Pixelii cu valori filtrate.
    """
    center = size // 2
    local_mean = np.mean(vec, dtype=np.float64)
    local_var = np.var(vec, dtype=np.float64)

    # Estimarea variantei zgomotului ca produs dintre constanta universala si media locala
    noise_var = Cu * local_mean

    # Calculul coeficientului sigma patrat (sigma^2)
    sigma_squared = local_var / (local_mean ** 2) if local_mean != 0 else 0

    if sigma_squared <= Cmax:
        weight = 1 - noise_var / (local_var if local_var != 0 else 1)  # Evitare impartire cu 0
        weight = np.clip(weight, 0, 1)  # Asigurarea valorii ponderii intre 0 si 1
    else:
        weight = 0

    # Calculul valorilor filtrate
    filtered_value = local_mean * weight + vec[center, center] * (1 - weight)
    return filtered_value


def apply_lee_sigma_filter(img, size, Cu, Cmax):
    h, w = img.shape
    capat = size // 2
    new_img = np.zeros((h, w), dtype=np.float64)

    for i in range(capat, h - capat):
        for j in range(capat, w - capat):
            vec = img[i - capat:i + capat + 1, j - capat:j + capat + 1]
            new_img[i, j] = lee_sigma_filter(vec, size, Cu, Cmax)

    # Handle the borders by copying edge values (simple method to avoid zero padding issues)
    new_img[:capat, :] = img[:capat, :]
    new_img[-capat:, :] = img[-capat:, :]
    new_img[:, :capat] = img[:, :capat]
    new_img[:, -capat:] = img[:, -capat:]

    return new_img



# p = ProductIO.readProduct('./subset_0_of_S1B_IW_GRDH_1SDV_20211003T135159_20211003T135224_028972_03751C_2107.dim')
#bands = list(p.getBandNames())
#print(bands)
#Bi = p.getBand(bands[0])
#w = Bi.getRasterWidth()
#h = Bi.getRasterHeight()
#Bi_data = np.zeros(w * h, np.float32)  # 16bits->img 12bits/band
#Bi.readPixels(0, 0, w, h, Bi_data)
#Bi_data.shape = h, w
#print(np.shape(Bi_data))
#print(np.min(Bi_data))
#print(np.max(Bi_data))
# h,_ = np.histogram(img, bins=2**16, range=(0, 2**16), density=True)
# print(h[0]) # probabilitati - hist
# print(h[1]) # limitele intervalelor
# plt.figure(), plt.plot(h), plt.show()
# img = abs(img)
# img = np.clip(img, 0, 2**16-1)
# plt.figure("Original"), plt.imshow(img, cmap='gray', norm=colors.LogNorm()), plt.colorbar(), plt.show()


# incarcare img satelitara
img = np.load('band-Intensity_VV.npy')[900:2580, 5400:7400]
print(np.shape(img))
print(np.min(img))
print(np.max(img))
print(type(img))

# afisare imagine originala
plt.figure(), plt.imshow(img, cmap='gray', norm=colors.LogNorm()), plt.colorbar()

# setare dimensiune filtru
filter_size = 3
capat = filter_size // 2

# aplicare si afisare filtru median
start = time.time()
img_lee = apply_lee_sigma_filter(img, filter_size, 0.54, 1)
stop = time.time()
print(f'Timp total:{stop-start} sec')
plt.figure("LeeSigma"), plt.imshow(img_lee, cmap='gray', norm=colors.LogNorm()), plt.colorbar(), plt.show()
