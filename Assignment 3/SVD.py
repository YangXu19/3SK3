import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

img = imread('McMasterLogo.png')

rgb_weights = [0.2989, 0.5879, 0.1140]

gray_img = np.dot(img[...,:3], rgb_weights)
# plt.imshow(gray_img, cmap=plt.get_cmap("gray"))

# SVD
U, S, VT = np.linalg.svd(gray_img, full_matrices=False)
S = np.diag(S)

j = 0
for r in (1, 2, 4, 16, 32, 64, 128, 220):
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