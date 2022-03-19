import math
import xlsxwriter
import scipy as sp
import random
import numpy as np

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
    return r_hat_nums

# Step 2
def r_hat_nums(Blist):

    r_hat_nums_noise = []

    # make random distribution values
    sp.random.seed(400186733)
    normal_dist = sp.random.normal(0, 0.2, 16)
    for i in range(16):
        r_hat_nums_noise.append(Blist[i] + normal_dist[i])

    # print(normal_dist)
    return r_hat_nums_noise

def Blist():
    B1 = r_hat_distances(0, 0.5, 0)
    B2 = r_hat_distances(3.5, 0, 1)
    B3 = r_hat_distances(0, 1.5, 0)
    B4 = r_hat_distances(4.5, 0, 0)
    B5 = r_hat_distances(0, 2.5, 3.5)
    B6 = r_hat_distances(1.5, 2.5, 0)
    B7 = r_hat_distances(3, 0, 2.0)
    B8 = r_hat_distances(4, 4, 4)
    return [B1, B2, B3, B4, B5, B6, B7, B8]

def Blist_noise():
    B1 = r_hat_distances(0, 0.5, 0)
    B2 = r_hat_distances(3.5, 0, 1)
    B3 = r_hat_distances(0, 1.5, 0)
    B4 = r_hat_distances(4.5, 0, 0)
    B5 = r_hat_distances(0, 2.5, 3.5)
    B6 = r_hat_distances(1.5, 2.5, 0)
    B7 = r_hat_distances(3, 0, 2.0)
    B8 = r_hat_distances(4, 4, 4)

# Step 2
    B1_noise = r_hat_nums(B1)
    B2_noise = r_hat_nums(B2)
    B3_noise = r_hat_nums(B3)
    B4_noise = r_hat_nums(B4)
    B5_noise = r_hat_nums(B5)
    B6_noise = r_hat_nums(B6)
    B7_noise = r_hat_nums(B7)
    B8_noise = r_hat_nums(B8)

    return [B1_noise, B2_noise, B3_noise, B4_noise, B5_noise, B6_noise, B7_noise, B8_noise]

def get_rows(somelist, num):
    row = []
    for i in somelist:
        row.append(i[num])
    return row

def main():
    blist = Blist()
    row1_b = get_rows(blist, 0)
    row2_b = get_rows(blist, 1)
    row3_b = get_rows(blist, 2)
    row4_b = get_rows(blist, 3)
    row5_b = get_rows(blist, 4)
    row6_b = get_rows(blist, 5)
    row7_b = get_rows(blist, 6)
    row8_b = get_rows(blist, 7)
    row9_b = get_rows(blist, 8)
    row10_b = get_rows(blist, 9)
    row11_b = get_rows(blist, 10)
    row12_b = get_rows(blist, 11)
    row13_b = get_rows(blist, 12)
    row14_b = get_rows(blist, 13)
    row15_b = get_rows(blist, 14)
    row16_b = get_rows(blist, 15)

    blist = Blist_noise()
    row1_bn = get_rows(blist, 0)
    row2_bn = get_rows(blist, 1)
    row3_bn = get_rows(blist, 2)
    row4_bn = get_rows(blist, 3)
    row5_bn = get_rows(blist, 4)
    row6_bn = get_rows(blist, 5)
    row7_bn = get_rows(blist, 6)
    row8_bn = get_rows(blist, 7)
    row9_bn = get_rows(blist, 8)
    row10_bn = get_rows(blist, 9)
    row11_bn = get_rows(blist, 10)
    row12_bn = get_rows(blist, 11)
    row13_bn = get_rows(blist, 12)
    row14_bn = get_rows(blist, 13)
    row15_bn = get_rows(blist, 14)
    row16_bn = get_rows(blist, 15)

    # for i in B8:
    #     print(i)

    # Step 3
    x_val = []
    y_val = []
    z_val = []

    # times in seconds
    t = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]

    # compute x, y, z coordinates for each time
    for num in t:
        if num <= 100:
            x = num * 40 / 1000
            y = 0
            z = 0
        if 101 < num <= 200:
            x = 4
            y = (num - 100) * 20 / 1000
            z = 0
        if 201 < num <= 300:
            x = 4
            y = 2
            z = (num - 200) * 5 / 1000
        x_val.append(x)
        y_val.append(y)
        z_val.append(z)

    xi = [0, 3.5, 0, 4.5, 0, 1.5, 3.0, 4.0]
    yi = [0.5, 0, 1.5, 0, 2.5, 2.5, 0, 4.0]
    zi = [0, 1, 0, 0, 3.5, 0, 2.0, 4.0]

    row1 = []
    row2 = []
    row3 = []

    JTf_r1 = []
    JTf_r2 = []
    JTf_r3 = []

    r1c1 = 0
    r1c2 = 0
    r1c3 = 0
    r2c1 = 0
    r2c2 = 0
    r2c3 = 0
    r3c1 = 0
    r3c2 = 0
    r3c3 = 0

    JTf_row1 = 0
    JTf_row2 = 0
    JTf_row3 = 0

    i = 0
    for j in range(7):
        r1c1 = r1c1 + ((x_val[i] - xi[j])**2)/((row1_b[j])**2)
        r1c2 = r1c2 + (((x_val[i] - xi[j])*(y_val[i] - yi[j])))/((row1_b[j])**2)
        r1c3 = r1c3 + (((x_val[i] - xi[j])*(z_val[i] - zi[j])))/((row1_b[j])**2)

        r2c1 = r2c1 + (((x_val[i] - xi[j])*(y_val[i] - yi[j])))/((row1_b[j])**2)
        r2c2 = r2c2 + ((y_val[i] - yi[j])**2)/((row1_b[j])**2)
        r2c3 = r2c3 + (((y_val[i] - yi[j])*(z_val[i] - zi[j])))/((row1_b[j])**2)

        r3c1 = r3c1 + (((x_val[i] - xi[j])*(z_val[i] - zi[j])))/((row1_b[j])**2)
        r3c2 = r3c2 + (((y_val[i] - yi[j])*(z_val[i] - zi[j])))/((row1_b[j])**2)
        r3c3 = r2c2 + ((z_val[i] - zi[j])**2)/((row1_b[j])**2)

        JTf_row1 = ((x_val[i] - xi[j])*(row1_b[j] - row1_bn[j]))/((row1_b[j]))
        JTf_row2 = ((y_val[i] - yi[j])*(row1_b[j] - row1_bn[j]))/((row1_b[j]))
        JTf_row3 = ((z_val[i] - zi[j])*(row1_b[j] - row1_bn[j]))/((row1_b[j]))

    row1.append(r1c1)
    row1.append(r1c2)
    row1.append(r1c3)

    row2.append(r2c1)
    row2.append(r2c2)
    row2.append(r2c3)

    row3.append(r3c1)
    row3.append(r3c2)
    row3.append(r3c3)

    JTf_r1.append(JTf_row1)
    JTf_r2.append(JTf_row2)
    JTf_r3.append(JTf_row3)

    JTJ_matrix = []
    JTJ_matrix.append(row1)
    JTJ_matrix.append(row2)
    JTJ_matrix.append(row3)

    np.array(JTJ_matrix)
    JTJ_matrix_transpose = np.array(JTJ_matrix)
    # print(JTJ_matrix_transpose)

    JTf_matrix = []
    JTf_matrix.append(JTf_r1)
    JTf_matrix.append(JTf_r2)
    JTf_matrix.append(JTf_r3)
    # print(JTf_matrix)

    matrix_dot = np.dot(JTJ_matrix_transpose, JTf_matrix)
    #print(matrix_dot)

    R_K = [[0], [0], [0]]
    for i in range(100):
        R_K = np.subtract(R_K, matrix_dot)

    print(R_K)

    # print(row1)
    # print(x_val)
    # print(y)
    # print(z)

if __name__ == '__main__':
    main()

