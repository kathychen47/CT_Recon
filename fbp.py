import pydicom
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft, fftshift
import logging

def arange2(start, stop=None, step=1):
    """#Modified version of numpy.arange which corrects error associated with non-integer step size"""
    if stop == None:
        a = np.arange(start)
    else:
        a = np.arange(start, stop, step)
        if a[-1] > stop-step:
            a = np.delete(a, -1)
    return a

def getProj(img,theta):
    numAngles = len(theta)
    sinogram = np.zeros((img.size[0], numAngles))
    # Iteratively calculate the integral value of each projection
    for n in range(numAngles):
        theta = np.linspace(0., 180., numAngles, endpoint=False)
        # rotate the image
        rotImgObj = img.rotate(90-theta[n], resample=Image.BICUBIC)
        # calculate the integral value of porjection
        sinogram[:,n] = np.sum(rotImgObj, axis=0)
    return sinogram

def filterProj(sinogram):
    projLen, numAngles = sinogram.shape
    step = 2 * np.pi / projLen
    w = arange2(-np.pi, np.pi, step)
    if len(w)<projLen:
        w = np.concatenate([w, [w[-1] + step]])
    ramp_filter = np.abs(w)
    ramp_filter = fftshift(ramp_filter)  # shifts the zero-frequency component of the discrete Fourier transform (DFT) to the center of the array

def filterProj2(sinogram,a=0.1):
    projLen, numAngles = sinogram.shape
    step = 2 * np.pi / projLen
    w = arange2(-np.pi, np.pi, step)
    if len(w) < projLen:
        w = np.concatenate([w, [w[-1] + step]])  # depending on image size, it might be that len(w) =
        # projLen - 1. Another element is added to w in this case
    rn1 = abs(2 / a * np.sin(a * w / 2));  # approximation of ramp filter abs(w) with a funciton abs(sin(w))
    rn2 = np.sin(a * w / 2) / (a * w / 2);  # sinc window with 'a' modifying the cutoff freqs
    ramp_filter = rn1 * (rn2) ** 2;
    ramp_filter = fftshift(ramp_filter)

    filter_sinogram = np.zeros_like(sinogram)
    for i in range(numAngles):
        projfft = fft(sinogram[:, i])  # one dimentional Fourier transform
        filter_projfft = projfft * ramp_filter  # *ramp_filter
        filter_proj = np.real(ifft(filter_projfft))  # inverse Fourier transform
        filter_sinogram[:, i] = filter_proj
    plt.imshow(filter_sinogram, cmap='gray')
    plt.title("filter_sinogram")
    plt.show()
    return filter_sinogram

def backPorj(filter_sinogram, theta):
    # define a empty reconstruction image
    N = filter_sinogram.shape[0]
    reconImg = np.zeros((N, N))  # the length of image is equal to the length of detector

    # Create a grid coordinate system with the center point at (0,0) in which the origin of the image coordinate system,
    # which is located at the upper left corner, is shifted to the center point.
    X = arange2(N) - N / 2  # [-N/2,...0..+N/2]
    Y = X.copy()
    x, y = np.meshgrid(X, Y)  # generate grid coordinate

    # Convert degrees to radians
    theta = theta * np.pi / 180

    numAngles = len(theta)
    for n in range(numAngles):
        t = x * np.sin(theta[n]) + y * np.cos(theta[n])  # x-y -> t-s
        # s = -x*np.sin(theta[n])+y*np.cos(theta[n])
        tCor = np.round(
            t + N / 2)  # Shift the coordinate axis origin to match the image origin. round() may result in floating-point numbers, needs to be rounded to an integer.
        tCor = tCor.astype("int")
        tIndex, sIndex = np.where((tCor >= 0) & (tCor <= N - 1))  # tIndex: row index; sIndex: column Index
        sino_angle_n = filter_sinogram[:, n]
        reconImg[tIndex, sIndex] += sino_angle_n[tCor[tIndex, sIndex]]
        # plt.imshow(reconImg, cmap='gray')
        # plt.title('Image {}'.format(n))
        # plt.show()
    reconImg = Image.fromarray((reconImg - np.min(reconImg)) / np.ptp(reconImg) * 255)  # normilization
    fig = plt.imshow(reconImg)
    plt.title('reconImg')
    plt.show()
    return reconImg

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

if __name__ == '__main__':

    # generate output logging
    logging.basicConfig(filename='output.log', level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # --------------------------------------------------Load data----------------------------------------------------------
    dcm = pydicom.dcmread('1.2.826.0.1.3680043.5876/1.dcm')
    img = Image.fromarray(dcm.pixel_array).convert('L')
    # img = np.asarray(img)
    # img = (img - np.min(img)) / np.ptp(img) * 255
    # img = img.astype(np.uint8)
    # img = Image.fromarray(img)

    logger.debug(f"Original_img: {img}")
    plt.imshow(img, cmap='gray')
    plt.title("original_image")
    plt.show()

    # -----------------------------------------------Forward Projection-----------------------------------------------------
    # Define rotation angle and the number of rotation
    numAngles = 720
    theta = np.linspace(0., 180., numAngles, endpoint=False)
    sinogram = getProj(img, theta)
    logger.debug(f"sinogram: {sinogram}")

    # -----------------------------------------------sinogram filteration---------------------------------------------------
    filter_sinogram=filterProj2(sinogram)
    logger.debug(f"filter_sinogram: {filter_sinogram}")

    # --------------------------------------------- Back Projection/ Recon -------------------------------------------------
    reconImg = backPorj(filter_sinogram, theta)
    logger.debug(f"reconImg: {reconImg}")

    # generate lineProfile
    lineProfile(200,img,reconImg)


    # Load logging
    with open('output.log', 'r') as f:
        content = f.readlines()
        print(content)

