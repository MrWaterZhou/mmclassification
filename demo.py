import cv2
import numpy as np
import sys


def fun1(image):
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    img = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return img


def fun2(image):
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(2, (8, 8))
    clahe.apply(image_yuv[:, :, 0])
    img = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
    return img


def imequalize(img):
    """Equalize the image histogram.

    This function applies a non-linear mapping to the input image,
    in order to create a uniform distribution of grayscale values
    in the output image.

    Args:
        img (ndarray): Image to be equalized.

    Returns:
        ndarray: The equalized image.
    """

    def _scale_channel(im, c):
        """Scale the data in the corresponding channel."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, (0, 255))[0]
        # For computing the step, filter out the nonzeros.
        nonzero_histo = histo[histo > 0]
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        if not step:
            lut = np.array(range(256))
        else:
            # Compute the cumulative sum, shifted by step // 2
            # and then normalized by step.
            lut = (np.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = np.concatenate([[0], lut[:-1]], 0)
        # If step is zero, return the original image.
        # Otherwise, index from lut.
        return np.where(np.equal(step, 0), im, lut[im])

    # Scales each channel independently and then stacks
    # the result.
    s1 = _scale_channel(img, 0)
    s2 = _scale_channel(img, 1)
    s3 = _scale_channel(img, 2)
    equalized_img = np.stack([s1, s2, s3], axis=-1)
    return equalized_img


def find_background(image):
    histo = np.histogram(image[:, :, 0], 256 // 2, (0, 255))[0]
    color_sort = np.argsort(histo)


    background_color = color_sort[-1]
    background_color_2 = color_sort[-2]
    # mask = ((image[:, :, 0] // 2 != background_color) * (image[:, :, 0] // 2 != background_color_2)).astype(np.uint8)
    mask = (image[:, :, 0] // 2 != background_color_2).astype(np.uint8)
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_size = 0
    max_image = image
    for contour in contours:
        if contour.shape[0] > 40:
            y, x, h, w = cv2.boundingRect(contour)
            if h*w > max_size:
                max_image = image[x:x + w, y:y + h]
                max_size = h*w

    return mask * 255


if __name__ == '__main__':
    file = 'b9389d11d63c43db93550505dda1dc58.jpg'
    image = cv2.imread(file)
    results = find_background(image)
    # print(results[2])
    i = 0
    for image_new in [results]:

        cv2.imwrite('demo_{}.jpg'.format(i), image_new)
        i += 1