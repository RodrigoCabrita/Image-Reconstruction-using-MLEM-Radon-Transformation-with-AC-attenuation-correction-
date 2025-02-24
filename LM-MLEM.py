import numpy as np
import matplotlib.pyplot as plt
import cv2

class Tissue:
    def __init__(self, density, attenuation):
        self.density = density  # g/cm³
        self.attenuation = attenuation  # cm^-1

class Bone(Tissue):
    def __init__(self):
        super().__init__(density=1.92, attenuation=0.573)  # Values for cortical bone at 60 keV

class SoftTissue(Tissue):
    def __init__(self):
        super().__init__(density=1.06, attenuation=0.2)  # Values for muscle at 60 keV

class Ar(Tissue):
    def __init__(self):
        super().__init__(density=0.1, attenuation=0)  # Values for muscle at 60 keV


class Phantom:
    def __init__(self, size=(400, 400)):
        self.size = size
        self.matrix = np.zeros(size)
        self.bone = Bone()
        self.soft_tissue = SoftTissue()
        self.ar = Ar()

    def add_circular_structure(self,h,k, a, b, tissue_type):
        y, x = np.ogrid[:self.size[0], :self.size[1]]
        #dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        ellipse = (x-h)**2/a**2 + (y-k)**2/b**2 <= 1
        self.matrix[ellipse] = tissue_type.attenuation

    def generate(self):
        # Add soft tissue background
        self.matrix.fill(self.ar.attenuation)

        # Add bone structures
        self.add_circular_structure(200,200,80 ,130, self.bone)
        self.add_circular_structure(200, 200, 70, 115, self.soft_tissue)

    def display(self):
        plt.imshow(self.matrix, cmap='gray')
        plt.colorbar(label='Attenuation coefficient (cm^-1)')
        plt.title('2D Phantom with Bone and Soft Tissue')
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.show()

# Create and display the phantom
phantom = Phantom()
phantom.generate()
phantom.display()