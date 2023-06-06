import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pydicom
import time


def lineProfile(height, img, reconImg):
    line_profile = np.array(img)[height, :]
    recon_line_profile = np.array(reconImg)[height, :]
    plt.plot(line_profile, label='Original Image')
    plt.plot(recon_line_profile, label='Reconstructed Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Intensity')
    plt.title('Line Profile at Pixel height= {}'.format(height))
    plt.legend()
    plt.show()
    return line_profile, recon_line_profile

def getProj(img, theta):
    numAngles = len(theta)
    sinogram = np.zeros((numAngles, img.size[1]))
    for n in range(numAngles):
        rotImgObj = img.rotate(theta[n], resample=Image.BICUBIC) # anticlockwise rotation
        sinogram[n, :] = np.sum(rotImgObj, axis=0) # sum of each column
    return sinogram

# Define SIRT algorithm
def sirt(img,A, b, num_iterations):
    """A is the system matrix
    b is the projection data
    C is the diagonal matrix of the column sums of A
    R is the diagonal matrix of the row sums of A
    X(t+1)=X(t)+C*(A^T)*R*(b-A*X(t))
    """
    rows, cols = img.size
    x = np.zeros((rows, cols))
    R = np.diag(1 / (np.sum(A, axis=1)+1e-10))
    C = np.diag(1 / (np.sum(A, axis=0)+1e-10))
    AT = A.T
    for i in range(num_iterations):
        residual = b - A.dot(x) #b(180,512) A(180,512) x(512,512)
        forward_projection = AT.dot(R.dot(residual))
        x += C.dot(forward_projection)
    return x

# Calculate system matrix A
def generate_system_matrix(img_size, detector_length, numAngles):
    # initialize system matrix A
    num_pixels = img_size * img_size
    A = np.zeros((detector_length * numAngles, num_pixels))

    # calculate the distance between two adjacent detector pixels
    detector_unit_size = float(detector_length) / img_size

    # loop over all angles
    for angle_index in range(numAngles):
        angle = angle_index * np.pi / numAngles # angle in radians

        # calculate the start and end points of the detector (the detector is centered at the center of image)
        start_x = -detector_length / 2
        end_x = detector_length / 2
        
        # calculate the direction vector of the scanning line
        direction_vector = np.array([np.cos(angle), np.sin(angle)])
        
        # loop over all pixels
        for pixel_index in range(num_pixels):
            # calculate the x and y coordinates of the current pixel
            pixel_x = pixel_index % img_size # %: the modulo operator
            pixel_y = pixel_index // img_size # //: floor division

            # calculate the position of the current pixel relative to the center of the image
            pixel_position = np.array([pixel_x - img_size / 2, pixel_y - img_size / 2])
            # print("pixel_position:", pixel_position)
            # distance is the projection of pixel_position onto direction_vector
            distance = np.dot(pixel_position, direction_vector)
     
            # check if the current pixel is within the detector range
            if distance >= start_x and distance <= end_x:
                # calculate the index of the detector pixel that corresponds to the current pixel
                detector_index = int((distance - start_x) / detector_unit_size)

                # set the corresponding element in the system matrix to 1
                A[angle_index * detector_length + detector_index, pixel_index] = 1.0
    return A


start_time = time.time()

# load image
dcm = pydicom.dcmread('1.2.826.0.1.3680043.5876/1.dcm')
img = Image.fromarray(dcm.pixel_array).convert('L')

# falten image
img = img.resize((128, 128))
img_flat1 = np.array(img).reshape((-1, 1))
img_flat = img_flat1.transpose()
img_flat = Image.fromarray(img_flat).convert('L')
print("flttened image size：", img_flat.size)

# get system matrix A
theta = np.linspace(0., 180., 180, endpoint=False)
numAngles = len(theta)
img_size= 128
detector_length= 128
numAngles = len(theta)
A = generate_system_matrix(img_size, detector_length, numAngles) # the size of A is (180*128, 128*128)

# get projection data
sinogram_flat = A.dot(img_flat1)

# Reconstruct image using SIRT algorithm
x = sirt(img_flat, A, sinogram_flat, num_iterations=150)

#reshape reconstructed image
x = np.reshape(x, (128, 128))

# Calculate PSNR
mse = np.mean((x - img)**2)
psnr = 10 * np.log10((255**2) / mse)
print("PSNR: {:.2f} dB".format(psnr))

# Plot original image and reconstructed image
plt.figure(figsize=(8, 4))
plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')
plt.subplot(132)
plt.imshow(x, cmap='gray')
plt.title("Reconstructed Image (PSNR: {:.2f} dB)".format(psnr))
plt.axis('off')
plt.show()

# Plot line profile
lineProfile(64, img, x)

# calculate running time
end_time = time.time()
run_time = end_time - start_time
print("running time：", run_time, "s")