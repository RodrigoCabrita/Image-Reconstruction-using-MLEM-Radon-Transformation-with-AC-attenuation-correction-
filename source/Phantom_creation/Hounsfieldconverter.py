import numpy as np
import matplotlib.pyplot as plt
#%%
def shepp_logan_3d(size_out=128, phantom_type="kak-slaney", get_ellipsoids = False):

    (ellipsoids, nVoxel, formula) = _parse_inputs(size_out, phantom_type)
    nVoxelX = nVoxel[2]
    nVoxelY = nVoxel[1]
    nVoxelZ = nVoxel[0]

    img_phantom = np.zeros(nVoxel)
    range_x = np.linspace(-1, +1, nVoxelX)
    range_y = np.linspace(-1, +1, nVoxelY)
    range_z = np.linspace(-1, +1, nVoxelZ)
    mesh_z, mesh_y, mesh_x = np.meshgrid(range_z, range_y, range_x, indexing='ij')

    mesh_x = mesh_x.reshape(-1)
    mesh_y = mesh_y.reshape(-1)
    mesh_z = mesh_z.reshape(-1)

    coord = np.vstack([mesh_z, mesh_y, mesh_x])
    img_phantom = img_phantom.reshape(-1)
    for ellipsoid in ellipsoids:
        asq   = ellipsoid[0]**2          # a^2
        bsq   = ellipsoid[1]**2          # b^2
        csq   = ellipsoid[2]**2          # c^2
        x0    = ellipsoid[3]             # x offset
        y0    = ellipsoid[4]             # y offset
        z0    = ellipsoid[5]             # z offset
        phi1  = ellipsoid[6]*np.pi/180   # 1st Euler angle in radians (rotation about z-axis)
        phi2  = ellipsoid[7]*np.pi/180   # 2nd Euler angle in radians (rotation about x'-axis)
        phi3  = ellipsoid[8]*np.pi/180   # 3rd Euler angle in radians (rotation about z"-axis)
        A     = ellipsoid[9]             # Amplitude change for this ellipsoid

        c1 = np.cos(phi1)
        s1 = np.sin(phi1)
        c2 = np.cos(phi2)
        s2 = np.sin(phi2)
        c3 = np.cos(phi3)
        s3 = np.sin(phi3)
        # Euler rotation matrix
        alpha = [   # Z      Y                    X
                 [     c2, -s2*c1         ,  s2*s1          ], # Z
                 [  c3*s2, -s3*s1+c3*c2*c1, -s3*c1-c3*c2*s1 ], # Y
                 [  s3*s2,  c3*s1+s3*c2*c1,  c3*c1-s3*c2*s1 ]  # X
                ]
        if formula==0:
            # Move the ellipsoid to the origin first, and rotate...
            coord_rot = np.dot(alpha, coord-np.array([[z0], [y0], [x0]]))
            idx = np.argwhere((coord_rot[2,:])**2/asq + (coord_rot[1,:])**2/bsq + (coord_rot[0,:])**2/csq <= 1)
            # Naive:
            # coord_rot = np.dot(alpha, coord-np.array([[z0], [y0], [x0]])) + np.array([[z0], [y0], [x0]])
            # idx = np.argwhere((coord_rot[2,:]-x0)**2/asq + (coord_rot[1,:]-y0)**2/bsq + (coord_rot[0,:]-z0)**2/csq <= 1)
        else:
            # (x0,y0,z0) rotates too!
            coord_rot = np.dot(alpha, coord)
            idx = np.argwhere((coord_rot[2,:]-x0)**2/asq + (coord_rot[1,:]-y0)**2/bsq + (coord_rot[0,:]-z0)**2/csq <= 1)
        img_phantom[idx] += A

    img_phantom = img_phantom.reshape(nVoxel)
    if get_ellipsoids:
        return img_phantom, ellipsoids
    else:
        return img_phantom


def _parse_inputs(size_out, phantom_type):
    """
    Returns:
     tuple (ellipsoids, nVoxel)
     * ellipsoids is the m-by-10 array which defines m ellipsoids,
       where m is 10 in the cases of the variants implemented in this file.
     * nVoxel is the 3 array which defines the number of voxels
    Parameters:
     phantom_type: One of {"kak-slaney", "yu-ye-wang", "toft-schabel"}
     size_out: An int or 3-vector.
       * int : the phantom voxel will be isotropic.
       * 3-vector: the size of the phantom image [nZ, nY, nX]
    """
    if type(size_out) == int:
        nVoxel = [size_out, size_out, size_out]
    elif (type(size_out) == list or type(size_out) == tuple) and len(size_out)==3:
        nVoxel = [size_out[0], size_out[1], size_out[2]]
    elif type(size_out)== np.ndarray and np.size(size_out)==3 :
        nVoxel = [size_out.reshape(-1)[0], size_out.reshape(-1)[1], size_out.reshape(-1)[2]]
    else:
        nVoxel = [128, 128, 128]

    if phantom_type == "kak-slaney":
        ellipsoids = kak_slaney()
        formula=0
    elif phantom_type == "yu-ye-wang":
        ellipsoids = yu_ye_wang()
        formula=0
    elif phantom_type == "toft-schabel":
        ellipsoids = toft_schabel()
        formula=1
    else:
        print(f"Unknown type {phantom_type}. yu-ye-wang is used.")
        ellipsoids = yu_ye_wang()
        formula=0

    return (ellipsoids, nVoxel, formula)


def kak_slaney():
    """
    The 3D Shepp-Logan head phantom. A is the relative density of water.
    Ref:
     [1] Kak AC, Slaney M, Principles of Computerized Tomographic Imaging, 1988. p.102
         http://www.slaney.org/pct/pct-errata.html
    """
    #            a        b        c      x0       y0      z0    phi1  phi2   phi3   A
    #        -------------------------------------------------------------------------
    ells = [[ 0.6900,  0.920,  0.900,  0.000,   0.000,   0.000,  0   ,  0,  0,  1.0],#elipse de tras representada a amarelo que corresponde ao osso do cranio
            [ 0.6624,  0.874,  0.880,  0.000,   0.000,   0.000,  0   ,  0,  0, -0.8],#elipse representada a azul que corresponde à materia cinzenta do cerebro
            [ 0.4100,  0.160,  0.210, -0.220,   0.000,  -0.250,  108 ,  0,  0, -0.2],#elipse esquerda
            [ 0.3100,  0.110,  0.220,  0.220,   0.000,  -0.250,  72  ,  0,  0, -0.2],#elipse direita
            [ 0.2100,  0.250,  0.500,  0.000,   0.350,  -0.250,  0   ,  0,  0,  0.2],#bola grande em baixo
            [ 0.0460,  0.046,  0.046,  0.000,   0.100,  -0.250,  0   ,  0,  0,  0.2],#bola pequena ao centro
            [ 0.0460,  0.023,  0.020, -0.080,  -0.650,  -0.250,  0   ,  0,  0,  0.1],#bola de cima esquerda
            [ 0.0460,  0.023,  0.020,  0.060,  -0.650,  -0.250,  90  ,  0,  0,  0.1],#bola de cima direita
            [ 0.0560,  0.040,  0.100,  0.060,  -0.105,   0.625,  90  ,  0,  0,  0.2],
            [ 0.0560,  0.056,  0.100,  0.000,   0.100,   0.625,  0   ,  0,  0, -0.2]]
    ells = np.asarray(np.matrix(ells))
    return ells

if __name__ == '__main__':
    from matplotlib import pyplot as plt

    nVoxelZ = 120  #256
    nVoxelY = 256 #128
    nVoxelX = 256 #64

    phantom0, e = shepp_logan_3d(size_out=[nVoxelZ,nVoxelY,nVoxelX], phantom_type="kak-slaney", get_ellipsoids=True)
    #phantom1    = shepp_logan_3d(size_out=[nVoxelZ,nVoxelY,nVoxelX], phantom_type="yu-ye-wang")
    #phantom2    = shepp_logan_3d(size_out=[nVoxelZ,nVoxelY,nVoxelX], phantom_type="toft-schabel")
    plt.imshow(np.concatenate(
                [phantom0[3*phantom0.shape[0]//8]],
                axis=1), cmap="Greys_r")
    plt.colorbar()
    plt.title("fantoma")
    plt.show()
    print(f"Ellipsoids of kak-slaney are\n{e}")

x=shepp_logan_3d(size_out=[nVoxelZ,nVoxelY,nVoxelX], phantom_type="kak-slaney", get_ellipsoids=True)



z=22