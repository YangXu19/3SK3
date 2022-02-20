import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

def main():
    img = imread('McMasterLogo.png')
    gray_img = grayscale(img)
    ranks = [1,2,6,16,32,64,128,220]
    svd_ranks(gray_img, ranks)

def grayscale(img):
    rgb_weights = [0.2989, 0.5879, 0.1140]
    gray_img = np.dot(img[...,:3], rgb_weights)
    return gray_img
    # plot gray image
    # plt.imshow(gray_img, cmap=plt.get_cmap("gray"))

def svd_ranks(gray_img, ranks):
    U, S, VT = np.linalg.svd(gray_img, full_matrices=False)
    S = np.diag(S)

    j = 0

    for r in ranks:
        Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
        # plt.figure(j+1)
        plt.subplot(5,5,j+1)
        j += 1
        img = plt.imshow(Xapprox)
        img.set_cmap('gray')
        plt.axis('off')
        plt.title('r = ' + str(r))

    plt.show()

# plt.ylabel('Singular Values')
# plt.title('McMasterLogo.png Singular Values')
#
# plt.plot(S)
# print(S)
# plt.show()

if __name__ == '__main__':
    main()