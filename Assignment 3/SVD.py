from turtle import shape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

def main():
    img = imread('lily.jpg')
    gray_img = grayscale(img)
    random_img = np.random.uniform(0,100,(220,220))
    ranks = [1,2,4,16,32,64,120]
    svd_ranks(gray_img, ranks)

def grayscale(img):
    rgb_weights = [0.2989, 0.5879, 0.1140]
    gray_img = np.dot(img[...,:3], rgb_weights)
    return gray_img

def svd_ranks(gray_img, ranks):
    U, S, VT = np.linalg.svd(gray_img, full_matrices=False)
    S = np.diag(S)

    # Plot Singular Values
    plt.ylabel('Singular Values')
    plt.title('McMasterLogo.png Singular Values')
    plt.plot(S)
    print(S)
    plt.show()

    j = 0

    for r in ranks:
        # Calculate Rank approximation and plot it
        Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
        plt.subplot(5,5,j+1)
        j += 1
        img = plt.imshow(Xapprox)
        img.set_cmap('gray')
        plt.axis('off')
        plt.title('r = ' + str(r))

        print("* RANK=" + str(r))  

        # Calculate Residual Percentage Error
        A_frob = np.linalg.norm(gray_img, 'fro')
        AminusB = np.subtract(gray_img,Xapprox)
        AminusB_frob = np.linalg.norm(AminusB, 'fro')
        Residual_Percentage_Error = (abs(AminusB_frob) / abs(A_frob)) * 100
        print("Residual Percentage Error = " + str(Residual_Percentage_Error))

        # Compression Rate of Approximation
        n = gray_img.shape[0]
        m = gray_img.shape[1]
        Compression_Rate = ((r+(r*m)+(r*n))/(n*m))
        print("Compression Rate = " + str(Compression_Rate))

    plt.show()

    




if __name__ == '__main__':
    main()