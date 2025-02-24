from struct import Struct
from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
import cv2


class SheppLoganPhantom:
    def __init__(self):
        self._boneStructure = Structure()
        self._boneStruture.setDensity(2.2)
        self._greyMass = Structure()
        self._greyMass.setDensity(1.04)
        self._CSF = Structure()
        self._CSF.setDensity(1.01)
        self._hematoma = Structure()
        self._hematoma.setDensity(1.1)
        self._ar.setForm(ellipse(150, 350, 20, 20))
        self._ar = Structure()
        self._ar.setDensity(0)
        self._ar.setForm(np.zeros(400,400))


    #def createmap:




class Structure:
    def __init__(self):
        self._density = None
        self._name = None
        self._form = None

    @property
    def density(self):
        return self._density

    def setDensity(self, density):
        self._density = density

    def form(self):
        return self._form

    def setForm(self, form):
        self._form = form




#0.4727

def hounsimg():
    image1=np.zeros((400, 400), dtype=np.float32)
    x=cv2.ellipse(image1,(200,200), (175,125), 90, 0, 360, 1,-1)#osso
    x=cv2.ellipse(x,(200,200), (165,115), 90, 0, 360, 0.7, -1)#massa cinzenta
    x=cv2.ellipse(x,(200,150), (40,40), 90, 0, 360, 0.45, -1)#bola CSF
    x=cv2.ellipse(x,(150,220), (70,30), 70, 0, 360, 0, -1)#bola ar grande esquerda
    x=cv2.ellipse(x,(250,220), (60,30), 110, 0, 360, 0, -1)#bola ar grande direita
    x=cv2.ellipse(x,(200,195), (10,10), 90, 0, 360, 0.47, -1)#bola materia branca pequena central
    x=cv2.ellipse(x,(200,225), (10,10), 90, 0, 360, 0.45, -1)#bola CSF pequena central abaixo
    x=cv2.ellipse(x,(180,330), (10,7), 0, 0, 360, 0.5, -1)#bola hematoma pequena em baixo esquerda
    x=cv2.ellipse(x,(200,330), (8,8), 0, 0, 360, 0.477, -1)#bola coágulo sangui pequena em baixo central
    hounsimg1=cv2.ellipse(x,(217,330), (10,7), 90, 0, 360, 0.46, -1)#bola abcesso pequena em baixo direita
    return hounsimg1

#0.098
#número de fotoes perdidos por cm de espessura
def attenmap():
    image1=np.zeros((400, 400), dtype=np.float32)
    x=cv2.ellipse(image1,(200,200), (175,125), 90, 0, 360, 0.151,-1)#osso
    x=cv2.ellipse(x,(200,200), (165,115), 90, 0, 360, 0.120, -1)#massa cinzenta
    x=cv2.ellipse(x,(200,150), (40,40), 90, 0, 360, 0.096, -1)#bola CSF
    x=cv2.ellipse(x,(150,220), (70,30), 70, 0, 360, 0.004, -1)#bola ar grande esquerda
    x=cv2.ellipse(x,(250,220), (60,30), 110, 0, 360, 0.004, -1)#bola ar grande direita
    x=cv2.ellipse(x,(200,195), (10,10), 90, 0, 360, 0.099, -1)#bola materia branca pequena central
    x=cv2.ellipse(x,(200,225), (10,10), 90, 0, 360, 0.096, -1)#bola CSF pequena central abaixo
    x=cv2.ellipse(x,(180,330), (10,7), 0, 0, 360, 0.097, -1)#bola hematoma pequena em baixo esquerda
    x=cv2.ellipse(x,(200,330), (8,8), 0, 0, 360, 0.097, -1)#bola coágulo sangui pequena em baixo central
    attenmap1=cv2.ellipse(x,(217,330), (10,7), 90, 0, 360, 0.1, -1)#bola abcesso pequena em baixo direita
    return attenmap1




def image_counts():
    num_events = 10000
    variation = 0.1
    mean_events = num_events / (1 + variation)
    image11=np.zeros((400, 400), dtype=np.float32)
    x1=cv2.ellipse(image11,(200,200), (200,200), 90, 0, 360,0,-1)#osso
    image_counts1=cv2.ellipse(x1,(200,200), (195,195), 90, 0, 360, 0.4, -1)#massa cinzenta
    for i in range(0,400):
        for j in range(0, 400):
            if image_counts1[i,j] > 0.1:
                image_counts1[i,j]= int(np.random.poisson(mean_events) * (1 + variation))

    return image_counts1

def attenuation_counts():

    image11=np.zeros((400, 400), dtype=np.float32)
    x1=cv2.ellipse(image11,(200,200), (200,200), 90, 0, 360,0,-1)#osso
    attenuation_counts1=cv2.ellipse(x1,(200,200), (195,195), 90, 0, 360, 0.096, -1)#agua

    return attenuation_counts1

def generate_poisson_numbers(n, center, variation=0.1):
    # Calculate lambda for Poisson distribution
    lambda_param = center

    # Generate Poisson distributed numbers
    numbers = np.random.poisson(lambda_param, n)


    scaled_numbers = center + (numbers - lambda_param) * (center * variation) / np.sqrt(lambda_param)

    # Calculate the range
    min_value = center * (1 - variation)
    max_value = center * (1 + variation)


    clipped_numbers = int(np.clip(scaled_numbers, min_value, max_value))

    return clipped_numbers


def shepp_logan_emission():
    file = "C:\\Users\\rodri\\OneDrive\\Área de Trabalho\\TESE\\output_file.txt"
    with open(file) as f:

        lines = f.readlines()

    lines = lines[0].split(";")
    lines1 = lines[0][17:]
    lines[0] = lines1
    lines[255] = lines[255][:-1]
    lines[256] = lines[256][:-1]
    image = np.zeros((256, 256))
    i = 0
    for line in lines:
        cols = line.split(" ")
        j = 0
        for col in cols:
            if not col == '':
                try:
                    image[i, j] = float(col)
                except ValueError:
                    cols[0] = col.split(";")
                    image[i, j] = float(col)
                j += 1

        i += 1
        print(col)

    n = len(image)

    # Step 1: Transpose the matrix
    for i in range(n):
        for j in range(i, n):
            image[i][j], image[j][i] = image[j][i], image[i][j]

    # Step 2: Reverse each row
    for i in range(n):
        image[i] = image[i][::-1]

    for i in range(256):
        for j in range(256):
            if  2.05> image[i,j] >1.6:
                image[i,j]= generate_poisson_numbers(1,10000)
            elif image[i,j] > 2.05:
                image[i, j] = generate_poisson_numbers(1,5000)
            elif 1.6>image[i,j]>-0.1 and 150>j>60 and 200>i>75:
                image[i, j] = generate_poisson_numbers(1,5000)
            elif 1.6> image[i,j] > 0:
                image[i, j] = generate_poisson_numbers(1,2000)
    return image


def shepp_logan_CT():
    file = "C:\\Users\\rodri\\OneDrive\\Área de Trabalho\\TESE\\outputgeek.txt"
    with open(file) as f:

        lines = f.readlines()

    lines = lines[0].split(";")
    lines1 = lines[0][15:]
    lines[0] = lines1
    lines[255] = lines[255][:-2]
    #lines[256] = lines[256][:-1]
    image = np.zeros((256, 256))
    i = 0
    for line in lines:
        cols = line.split(" ")
        j = 0
        for col in cols:
            if not col == '':
                try:
                    image[i, j] = float(col)
                except ValueError:
                    cols[0] = col.split(";")
                    image[i, j] = float(col)
                j += 1

        i += 1
        print(col)
        #image=np.rot90(image)
    n = len(image)

    # Step 1: Transpose the matrix
    for i in range(n):
        for j in range(i, n):
            image[i][j], image[j][i] = image[j][i], image[i][j]

    # Step 2: Reverse each row
    for i in range(n):
        image[i] = image[i][::-1]

    for i in range(256):
        for j in range(256):
            if  2.05> image[i,j] >1.11:
                image[i, j] = generate_poisson_numbers(1,1818)
            elif  image[i,j] ==1.02:
                image[i, j] = generate_poisson_numbers(1,100)
            elif  1.041>image[i,j] >1.02:
                image[i, j] = generate_poisson_numbers(1,150)
            elif  1.02>image[i,j] >0.9:
                image[i, j] = generate_poisson_numbers(1,150)
            elif  0.9>image[i,j] :
                ger_neg = generate_poisson_numbers(1,900)
                image[i,j]= -ger_neg
    backgr_max=[]
    for l in range(256):
        for o in range(256):
            if image[l,o]>0:
                image[l,o] = (9.05*(10**-5)*image[l,o])+0.154
            else:
                coff= (1.54*(10**-4)*image[l,o])+0.154
                image[l,o] = coff
                backgr_max.append(coff)


    return image, backgr_max



# tyr=shepp_logan_emission()
# [thor, backgr_max]=shepp_logan_CT()
# plt.imshow(thor, cmap='gray')
#
# plt.show()
# udy=np.max(backgr_max)
#
# tt=9

