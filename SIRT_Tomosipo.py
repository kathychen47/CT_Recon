import tomosipo as ts
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from PIL import Image

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


vg = ts.volume(shape=(1, 128, 128)) # generate 1 volume geometry with size 512*512
pg = ts.parallel(angles=180, shape=(1, 725)) # generate 1 parallel geometry with 180 angles and 725 rays per angle

A = ts.operator(vg, pg) # forward projection, generate system matrix
# A is the system matrix from volume to projection. we should get a image size to create A.
# load image
print(A.domain_shape)

dcm = pydicom.dcmread('1.2.826.0.1.3680043.5876/1.dcm')
img = Image.fromarray(dcm.pixel_array).convert('L') # convert to grayscale
img = img.resize((128, 128))
x = np.array(img) # convert to numpy array
x = np.reshape(x, (1, 128, 128)) # reshape to 1*512*512
y = A(x) # A(x) is A*x, forward projection
# plt.imshow(y[:, 0, :])  # first projection
# plt.imshow(y[:, 8, :])  # quarter rotation
# plt.show()


R = 1 / (A(np.ones(A.domain_shape, dtype=np.float32))+ts.epsilon) # add epsilon to avoid division by zero
R = np.minimum(R, 1 / ts.epsilon) # avoid division by zero # R is the reciprocal of the sum of the rows of A
C = 1 / (A.T(np.ones(A.range_shape, dtype=np.float32))+ts.epsilon)
C = np.minimum(C, 1 / ts.epsilon)


num_iterations = 150
x_rec = np.zeros(A.domain_shape, dtype=np.float32)

# Print shape of x_rec, A(x_rec), and y
print(A.domain_shape)

for i in range(num_iterations):
    x_rec += C * A.T(R * (y - A(x_rec)))

# Evaluate reconstruction performance (PSNR)
mse = np.mean((x - x_rec)**2)
psnr = 10 * np.log10((255**2) / mse)
print("PSNR: {:.2f} dB".format(psnr))

x_rec = np.reshape(x_rec, (128, 128))
plt.subplot(121)
plt.imshow(img, cmap='gray')  # central slice of reconstruction
plt.title("Original Image")
plt.subplot(122)
plt.imshow(x_rec, cmap='gray')  # central slice of reconstruction
plt.title("Reconstructed Image (PSNR: {:.2f} dB)".format(psnr))
plt.show()

svg = ts.svg(vg, pg)
svg.save("intro_forward_projection_geometries.svg")

lineProfile(64, img, x_rec)