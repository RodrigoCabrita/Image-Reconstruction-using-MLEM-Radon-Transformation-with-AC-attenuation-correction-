import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
from source.Phantom_creation import hounsimg, attenmap, image_counts, attenuation_counts, shepp_logan_emission, shepp_logan_CT
from scipy import ndimage
from skimage.filters import unsharp_mask
from scipy.ndimage import gaussian_filter
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim
from tabulate import tabulate
import cv2


def reconstruction_MLEM(image1, mpatten1, step, iter):
    plt.ion()
    #Carregar fantomagit init

    #mpatten = shepp_logan_CT()
    #image = image_counts()
    #image = shepp_logan_phantom()
    activity_level=1;#controlar a concentração radioativa, neste caso o valor máximo vai ser menor, mais parecido com a imagem PET
    image = image1
    minimononzero=np.min(image[image!=0])
    maximononzero = np.max(image[image != 0])
    [mpatten,backgr_max]= mpatten1
    #Definição subplor, [0,0] indica a representação da imagem original no primeiro slot
    #fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    #axs[0, 0].imshow(image, cmap="Greys_r", vmin=minimononzero, vmax=maximononzero);    axs[0, 0].set_title("Original")

    #Obter sinograma
    #Faz-se apenas 180 graus e não 360, uma vez que iria-mos obter uma função par, ou seja, a mesma informação apartir dos 180 graus
    theta = np.linspace(0.0, 180.0, step, endpoint=False)#vai dos 0 aos 180 graus e tem que fazer 180 steps, ou seja, vai ter 180 ângulos de projeção
    sinogram = radon(image, theta, circle=False)
    #axs[0, 1].imshow(sinogram.T, cmap="Greys_r");    axs[0, 1].set_title("Sinograma")#O .T roda 90º a imagem


    #De modo a iniciar o algoritmo de reconstrução é necessário uma imagem inicial que depois vai sendo atualizada ao longo das iterações para este efeito cria-se uma matriz só de uns do mesmo tamanho da imagem original
    mlem_rec = np.ones(image.shape)#ponto de partida
    mlem_rec_sense=np.ones(image.shape)

    #Sinograma de uns
    sino_ones = np.ones(sinogram.shape)
    #sens_image = mpatten

    #atten_sin= radon(mpatten, theta, circle=None)
    #Imagem de sensibilidade, Back projection da matriz do sinograma de uns
    sens_image_ones = iradon(sino_ones, theta, circle=False, filter_name=None)
    mpatten[mpatten <= np.max(backgr_max)] = 0
    #sens_image1 = radon(mpatten, theta, circle=False)
    #sens_image=sino_ones*sens_image1
    #sens_image=iradon(sens_image, theta, circle=False)
    sens_image=mpatten

    fw=radon(sens_image, theta, circle=False)
    #sens_image[sens_image>0]=sens_image[sens_image>0]*sens_image_ones[sens_image>0]
    #sens_image = unsharp_mask(sens_image, radius=1, amount=1)

    #sens_image = sens_image / np.max(sens_image)  # normalization
    #normalization_factor=(x*15)/10000



    correction=np.zeros(sens_image.shape)
    correction_radon2=np.zeros(sens_image.shape)
    for i in range(iter):

        fp_com_corre = radon(mlem_rec_sense, theta, circle=False)
        fp_sem_corre = radon(mlem_rec, theta, circle=False)#forward projection do algoritmo mlem na iteração k (A x X^k)
        sinogram1=sinogram*fw
        sinogram2=sinogram*sino_ones
        #ratio sinograma que equivale a um sinograma simulado
        ratio_com_corre = np.copy(sinogram)
        ratio_com_corre[fp_com_corre!= 0] = sinogram1[fp_com_corre!= 0] / fp_com_corre[fp_com_corre!= 0] #(m/A x X^k)
        ratio_sem_corre = np.copy(sinogram)
        ratio_sem_corre[fp_sem_corre != 0] = sinogram2[fp_sem_corre != 0] / fp_sem_corre[fp_sem_corre != 0]  # (m/A x X^k)
        #Imagem de correção, inverso da transformada de radon (Back projection do ratio)

        #back-projection
        coeficiente_com_corre= iradon(ratio_com_corre, theta, circle=False, filter_name=None)
        coeficiente_sem_corre = iradon(ratio_sem_corre, theta, circle=False, filter_name=None)

        correction_radon2 = coeficiente_sem_corre / iradon(sino_ones, theta, circle=False, filter_name=None)
        #correction = coeficiente_com_corre/iradon(fw, theta, circle=False, filter_name=None)
        ax=iradon(fw, theta, circle=False, filter_name=None)
        correction[ax>0] = coeficiente_com_corre[ax>0] / ax[ax>0]

        mlem_rec *= correction_radon2
        mlem_rec_sense *=  correction

    #mlem_rec_sense=mlem_rec_sense*(10000/np.mean(mlem_rec_sense[50:72,70:100]))
    #mlem_rec = mlem_rec * (10000/np.mean(mlem_rec[50:72,70:100]))
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        axs[0, 0].imshow(correction_radon2, cmap="Greys_r")
        axs[0, 0].set_title('coeficientes de ajuste')
        axs[0, 1].imshow(mlem_rec, cmap="Greys_r")
        axs[0, 1].set_title('reconstrução sem sense')
        axs[1, 0].imshow(correction, cmap="Greys_r")
        axs[1, 0].set_title('coeficiente de ajuste com sensebilidade')
        axs[1, 1].imshow(mlem_rec_sense,cmap="Greys_r")
        axs[1, 1].set_title('reconstrução com sense')
        axs[1, 2].imshow(image, cmap="Greys_r")
        axs[1, 2].set_title('imagem original')
        axs[0, 2].imshow(sinogram1, cmap="Greys_r")
        axs[0, 2].set_title('sensibilidade')
        #axs[1, 1].set_colorbar()
        fig.suptitle('mlem recon image it=%d' % (i + 1), fontsize=30)
        plt.show()
    ty=90
    #axs[1, 1].imshow(fp.T, cmap="Greys_r");         axs[1, 1].set_title("FP")
    #axs[0, 2].imshow(ratio.T, cmap="Greys_r");      axs[0, 2].set_title("Ratio")
    #axs[1, 2].imshow(sens_image, cmap="Greys_r");   axs[1, 2].set_title("sense")
    #axs[1, 0].imshow(mlem_rec, cmap='Greys_r');   axs[1, 0].set_title('mlem recon image it=%d' % (i+1))

    #plt.show()

    return image, mlem_rec, mlem_rec_sense, correction, correction_radon2 ,sens_image, fw, sinogram1


[image, mlem_rec, mlem_rec_sense, correction, correction_radon2, sens_image, fw, sinogram1]=reconstruction_MLEM(shepp_logan_emission(),shepp_logan_CT(), 180, 10)

#3 bolinhas de baixo
#normalized_shepp_emission=shepp_logan_emission()[190:215,115:145]/np.max(shepp_logan_emission()[190:215,115:145])
#normalized_mlem_rec_sense=mlem_rec_sense[190:215,115:145]/np.max(shepp_logan_emission()[190:215,115:145])
#normalized_mlem_rec=mlem_rec[190:215,115:145]/np.max(shepp_logan_emission()[190:215,115:145])

#restantes
normalized_shepp_emission=shepp_logan_emission()[10:245,40:215]/np.max(shepp_logan_emission()[10:245,40:215])
normalized_mlem_rec_sense=mlem_rec_sense[10:245,40:215]/np.max(shepp_logan_emission()[10:245,40:215])
normalized_mlem_rec=mlem_rec[10:245,40:215]/np.max(shepp_logan_emission()[10:245,40:215])

def PSNR(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

MSE1=np.square(np.subtract(normalized_shepp_emission,normalized_mlem_rec_sense)).mean()
RMSE1=np.sqrt(MSE1)
PSNR1=PSNR(normalized_shepp_emission,normalized_mlem_rec_sense)
SSIM1 = ssim(normalized_shepp_emission, normalized_mlem_rec_sense, data_range=normalized_shepp_emission.max() - normalized_shepp_emission.min())
MSE2=np.square(np.subtract(normalized_shepp_emission,normalized_mlem_rec)).mean()
RMSE2=np.sqrt(MSE2)
PSNR2=PSNR(normalized_shepp_emission,normalized_mlem_rec)
SSIM2 = ssim(normalized_shepp_emission, normalized_mlem_rec, data_range=normalized_shepp_emission.max() - normalized_shepp_emission.min())

head = [" ", "Reconstrução com correção", "Reconstrução sem correção"]

mydata = [
    ["MSE", MSE1, MSE2],
    ["RMSE", RMSE1, RMSE2],
    ["PSNR", PSNR1, PSNR2],
    ["SSIM", SSIM1, SSIM2]
]

print(tabulate(mydata, headers=head, tablefmt="grid"))

backend_orig = plt.get_backend()

while True:
    ent = input("Press the Enter key to continue.")
    if ent == "":
        break

i=True
while i==True:
    matplotlib.use('Qt5Agg')
    plt.figure()
    plt.imshow(mlem_rec_sense, cmap="Greys_r")
    coords = plt.ginput(n=1, timeout=0)  # n=-1 for unlimited points until Enter is pressed
    print("Selected coordinates:", coords)
    plt.show()
    plt.colorbar()
    if ent == "":
        i=False
        plt.switch_backend(backend_orig)
        plt.close()


while True:
    ent = input("Press the Enter key to continue.")
    if ent == "":
        break


[tyr,thor]=coords[0]
yt=mlem_rec[int(thor),:]
yt=yt/np.max(yt)
filter = gaussian_filter(yt, sigma=4)
plt.figure()
plt.plot(yt)
plt.plot(filter)
plt.show()


yy=8
