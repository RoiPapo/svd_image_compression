from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_rank


def K_estimation(mat, k, n, m):
    U, s, Vt = np.linalg.svd(mat)
    Sigma = np.zeros((m, n))
    for i in range(s.size):
        Sigma[i, i] = s[i]

    approx = U @ Sigma[:, :k] @ Vt[:k, :]
    return approx


def main():
    img = Image.open("pic1.jpg")
    print(img.format, img.size, img.mode)
    k = 40
    n = img.size[0]
    m = img.size[1]
    pix = np.array(img)

    red_array = pix[:, :, 0]
    green_array = pix[:, :, 1]
    blue_array = pix[:, :, 2]
    KredARR = K_estimation(red_array, k, n, m)
    KgreenARR = K_estimation(green_array, k, n, m)
    KblueARR = K_estimation(blue_array, k, n, m)

    constructedMat = np.dstack((KredARR, KgreenARR, KblueARR))
    image_k = Image.fromarray(np.uint8(constructedMat))
    plt.imshow(image_k, cmap="gray")
    mone = np.linalg.norm(red_array - KredARR, "fro") ** 2
    mehane = np.linalg.norm(red_array, "fro") ** 2
    print(mone / mehane)

    plt.show()


if __name__ == "__main__":
    main()
