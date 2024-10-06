import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
import cupyx.scipy.ndimage


def gpu_lee_sigma_filter(img, size, Cu, Cmax):
    img_gpu = cp.asarray(img)  # Transferul imaginii catre GPU
    h, w = img_gpu.shape
    capat = size // 2

    # Pregatirea variabilei de iesire cu aceeasi dimensiune ca imaginea input
    new_img_gpu = cp.zeros((h, w), dtype=img_gpu.dtype)

    # Definirea nucleului de mediere (filtru uniform)
    kernel = cp.ones((size, size), dtype=cp.float32) / (size * size)

    # Aplic bordarea imaginii pentru a gestiona marginile: reflect, constant, nearest, mirror sau wrap
    img_padded = cp.pad(img_gpu, pad_width=capat, mode='reflect')

    # Calculul mediei locale si a mediei patratice locale folosind convolutia
    local_mean = cupyx.scipy.ndimage.convolve(img_padded, kernel, mode='reflect')[capat:-capat, capat:-capat]
    local_sqr_mean = cupyx.scipy.ndimage.convolve(img_padded**2, kernel, mode='reflect')[capat:-capat, capat:-capat]
    local_var = local_sqr_mean - local_mean**2

    # Calculul variantei zgomotului si a coeficientului de variatie patrat (simga^2)
    noise_var = Cu * local_mean
    sigma_squared = local_var / (local_mean ** 2 + 1e-10)  # Avoid division by zero

    # Calculul ponderilor
    weight = cp.where(sigma_squared <= Cmax, 1 - noise_var / cp.maximum(local_var, 1e-10), 0)
    weight = cp.clip(weight, 0, 1)  # Ensure the weight is between 0 and 1

    # Calculul filtrarii imaginii cu ponderile rezultate
    new_img_gpu = local_mean * weight + img_gpu * (1 - weight)

    # Transferul imaginii inapoi catre CPU
    return cp.asnumpy(new_img_gpu)


# incarcarea imaginii satelitare
img = np.load('band-Intensity_VV.npy')[1000:1960, 6000:7600]
img = np.ascontiguousarray(img)
print(np.shape(img))
print(np.min(img))
print(np.max(img))
print(type(img))

# afisare imagine originala
#plt.figure(), plt.imshow(img, cmap='gray', norm=colors.LogNorm()), plt.colorbar()

# conversie nivele de gri si gama 0-255, uint 8
# img = (color.rgb2gray(img) * 255).astype(cp.uint8)

# afisare imagine originala
# plt.figure("Original"), plt.imshow(img, cmap='gray')

# hist, bins = cp.histogram(img, bins=4096, density=True)

# plt.figure('Hist'), plt.bar(bins[:-1], hist), plt.show()

# aplica zgomot
# img_noise = add_impulsive_noise(img, 0.95)

# afiseaza imaginea cu zgomot
# plt.figure("Zgomotos"), plt.imshow(img_noise, cmap='gray'), plt.show()
# afiseaza MSE imagine cu zgomot
# print('MSE imagine cu zgomot: {}'.format(mse(img, img_noise, 0)))

# setare dimensiune filtru
filter_size = 7
capat = filter_size // 2

# aplicare si afisare filtru median
start = time.time()
img_lee = gpu_lee_sigma_filter(img, filter_size, 0.54, 1)
stop = time.time()
print(f'Timp total:{stop-start} sec')
#plt.figure("LeeSigma"), plt.imshow(img_lee, cmap='gray', norm=colors.LogNorm()), plt.colorbar(), plt.show()
print(cp.min(img_lee))
print(cp.max(img_lee))
