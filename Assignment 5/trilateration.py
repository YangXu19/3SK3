import math
import xlsxwriter
import scipy as sp
import random

# step 1
def r_hat_distances(xi, yi, zi):
    r_hat_nums = []

    # times in seconds
    t = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

    # compute x, y, z distances in km
    for num in t:
        if num <= 100:
            x = num*40/1000
            y = 0
            z = 0
        if 101 < num <= 200:
            x = 4
            y = (num - 100)*20/1000
            z = 0
        if 201 < num <= 300:
            x = 4
            y = 2
            z = (num - 200)*5/1000

        # compute distance in m
        r_hat_nums.append(math.sqrt(((x - xi) ** 2) + ((y - yi) ** 2) + ((z - zi) ** 2)) * 1000)
    #return r_hat_nums

    # Step 2
    r_hat_nums_noise = []

    # make random distribution values
    sp.random.seed(400186733)
    normal_dist = sp.random.normal(0, 0.2, 16)
    for i in range(16):
        r_hat_nums_noise.append(r_hat_nums[i] + normal_dist[i])

    #print(normal_dist)
    return r_hat_nums_noise

def main():

    B1 = r_hat_distances(0, 0.5, 0)
    B2 = r_hat_distances(3.5, 0, 1)
    B3 = r_hat_distances(0, 1.5, 0)
    B4 = r_hat_distances(4.5, 0, 0)
    B5 = r_hat_distances(0, 2.5, 3.5)
    B6 = r_hat_distances(1.5, 2.5, 0)
    B7 = r_hat_distances(3, 0, 2.0)
    B8 = r_hat_distances(4, 4, 4)

    for i in B8:
        print(i)

if __name__ == '__main__':
    main()