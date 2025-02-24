
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform.radon_transform import _get_fourier_filter
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale


#Carregar fantoma e a sua respetiva representação em sinograma
image = shepp_logan_phantom()
image = rescale(image, scale=0.7, mode='reflect', channel_axis=None)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta)
dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(
    sinogram,
    cmap=plt.cm.Greys_r,
    extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    aspect='auto',
)

fig1.tight_layout()
plt.show()

#%%
#Display dos diferentes tipos de filtros que podem ser utilizados no algoritmo Codigo_1
filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']

for ix, f in enumerate(filters):
    response = _get_fourier_filter(2000, f)
    plt.plot(response, label=f)

plt.xlim([0, 2000])
plt.xlabel('frequency')
plt.legend()
plt.show()

#%%
#Aplicação do Codigo_1
from skimage.transform import iradon

reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='cosine')
#CALCULO DO ERRO
error = reconstruction_fbp - image
print(f'FBP rms reconstruction error: {np.sqrt(np.mean(error**2)):.3g}')

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction error\nFiltered back projection")
ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()

