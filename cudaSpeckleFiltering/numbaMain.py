from numba import cuda
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors

@cuda.jit
def lee_sigma_filter_kernel(d_img, d_result, size, Cu, sigma):
    x, y = cuda.grid(2)
    if x >= d_img.shape[0] or y >= d_img.shape[1]:
        return  # Verifică dacă indexul este în afara limitelor imaginii

    # Definirea ferestrei
    half_size = size // 2
    start_x = max(x - half_size, 0)
    end_x = min(x + half_size + 1, d_img.shape[0])
    start_y = max(y - half_size, 0)
    end_y = min(y + half_size + 1, d_img.shape[1])

    # Calculul mediei și varianței locale
    local_sum = 0.0
    local_sum_sq = 0.0
    pixel_count = 0

    for i in range(start_x, end_x):
        for j in range(start_y, end_y):
            pixel_val = d_img[i, j]
            local_sum += pixel_val
            local_sum_sq += pixel_val ** 2
            pixel_count += 1

    if pixel_count > 0:
        local_mean = local_sum / pixel_count
        local_var = (local_sum_sq / pixel_count) - (local_mean ** 2)
        noise_var = Cu * local_mean

        # Calculul coeficientului de variație (sigma^2)
        if local_mean != 0:
            sigma_squared = local_var / (local_mean ** 2)
        else:
            sigma_squared = 0

        # Calculul ponderii
        if sigma_squared <= sigma:
            weight = 1 - noise_var / max(local_var, 1e-6)  # Evită împărțirea la zero
            weight = max(min(weight, 1), 0)  # Asigură că greutatea este între 0 și 1
        else:
            weight = 0

        # Aplicarea filtrului
        d_result[x, y] = local_mean * weight + d_img[x, y] * (1 - weight)


# Inițializarea și rularea kernelului
def apply_lee_sigma_filter(img, size, Cu, sigma):
    # Prepară datele și configurația kernelului
    d_img = cuda.to_device(img)
    d_result = cuda.device_array_like(img)

    threadsperblock = (16, 16)
    blockspergrid_x = (img.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
    blockspergrid_y = (img.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print(f"Threads per block: {threadsperblock}")
    print(f"Blocks per grid: {blockspergrid}")

    # Lansează kernelul
    lee_sigma_filter_kernel[blockspergrid, threadsperblock](d_img, d_result, size, Cu, sigma)

    # Copiază rezultatul înapoi pe CPU și returnează-l
    result = d_result.copy_to_host()
    return result


# incarcarea imaginii satelitare
img = np.load('band-Intensity_VV.npy')[1000:1960, 6000:7600] # [1900:2500, 6600:7400], [1500:2500, 6400:7400],
                                                            # [1500:2500, 5900:7400][900:2580, 5400:7400], [1000:2600, 6000:7980]
img = np.ascontiguousarray(img) # pas aditional CUDA
    # toata imaginea in aceeasi zona de memorie
print(np.shape(img))
print(np.min(img))
print(np.max(img))
print(type(img))

# afisare imagine originala
plt.figure(), plt.imshow(img, cmap='gray', norm=colors.LogNorm()), plt.colorbar()


# setare dimensiune filtru
filter_size = 7
capat = filter_size // 2

# aplicare si afisare filtru median
start = time.time()
img_lee = apply_lee_sigma_filter(img, filter_size, 0.54, 1)
stop = time.time()
print(f'Timp total:{stop-start} sec')
plt.figure("LeeSigma"), plt.imshow(img_lee, cmap='gray', norm=colors.LogNorm()), plt.colorbar(), plt.show()

