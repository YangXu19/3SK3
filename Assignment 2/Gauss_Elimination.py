import numpy
import time
from numpy import array, zeros, random


def Gauss_Elimination(A_mat, b_vec):
    # check that A_mat is square
    row = len(A_mat)
    col = len(A_mat[0])

    if row != col:
        print("A_mat is NOT square!")
        # return zero vector

    # check that b_vec has the appropriate number of elements
    b_row = len(b_vec)

    if b_row != row:
        print("b_vec does not have the correct number of rows!")
        # return zero vector

    ut_mat, new_b_vec = Forward_Elimination(A_mat, b_vec)
    Back_Substitution(ut_mat, new_b_vec)


# FORWARD ELIMINATION
def Forward_Elimination(A_mat, b_vec):
    # number of linear equations
    n = len(b_vec)
    for column in range(n-1):
        for row in range(column+1, n):
            # if element to eliminate is already 0
            if A_mat[row, column] == 0:
                continue
            factor = A_mat[column, column]/A_mat[row, column]
            for i in range(column, n):
                # elimination statement
                A_mat[row, i] = A_mat[column, i] - (A_mat[row, i]*factor)

            # new b vector
            b_vec[row] = b_vec[column] - b_vec[row]*factor

    print("A_mat is " + str(A_mat))
    print("b_vec is " + str(b_vec))

    return A_mat, b_vec


# BACK SUBSTITUTION
def Back_Substitution(ut_mat, new_b_vec):
    n = len(new_b_vec)
    # create solution vector
    solu_vec = zeros(n, float)

    solu_vec[n-1] = new_b_vec[n-1]/ut_mat[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i+1, n):
            sum += ut_mat[i, j]*solu_vec[j]
        solu_vec[i] = (new_b_vec[i] - sum)/ut_mat[i, i]

    print("The solution of the system of linear equations is: " + str(solu_vec))


def Gauss_Elimination_Pivoting(A_mat, b_vec):
    # check that A_mat is square
    row = len(A_mat)
    col = len(A_mat[0])

    if row != col:
        print("A_mat is NOT square!")
        # return zero vector

    # check that b_vec has the appropriate number of elements
    b_row = len(b_vec)

    if b_row != row:
        print("b_vec does not have the correct number of rows!")
        # return zero vector

    ut_mat, new_b_vec = Forward_Elimination_Pivoting(A_mat, b_vec)
    Back_Substitution(ut_mat, new_b_vec)


def swap_rows(mat, vec, row1, row2):
    mat[[row1, row2]] = mat[[row2, row1]]
    vec[[row1, row2]] = mat[[row2, row1]]

# With pivoting
def Forward_Elimination_Pivoting(A_mat, b_vec):
    # number of linear equations
    n = len(b_vec)
    for column in range(n - 1):
        # swap rows if 0
        if numpy.fabs(A_mat[column, column]) < 1.0e-12:
            for row in range(column + 1, n):
                if numpy.fabs(A_mat[row, column]) > numpy.fabs(A_mat[column, column]):
                    swap_rows(A_mat, b_vec, column, row)
                    break

        for row in range(column + 1, n):
            # if element to eliminate is already 0
            if A_mat[row, column] == 0:
                continue
            factor = A_mat[column, column] / A_mat[row, column]
            for i in range(column, n):
                # elimination statement
                A_mat[row, i] = A_mat[column, i] - (A_mat[row, i] * factor)

            # new b vector
            b_vec[row] = b_vec[column] - b_vec[row] * factor

    print("A_mat with pivoting is " + str(A_mat))
    print("b_vec with pivoting is " + str(b_vec))

    return A_mat, b_vec


# Generate random test case with an n x n coefficient matrix and an n x 1 right-hand side vector filled with random values from -100 to 100
def random_test_case(n):
    A_mat = array(random.uniform(-100.0, 100.0, (n,n)), float)
    b_vec = array(random.uniform(-100.0, 100.0, n), float)
    return A_mat, b_vec

# Generate random test cases with an n x n coefficient matrix and an n x 1 right-hand side vector filled with random values specified in range
def random_test_case_in_range(n, low, high):
    A_mat = array(random.uniform(low, high, (n,n)), float)
    b_vec = array(random.uniform(low, high, n), float)
    return A_mat, b_vec


def main():
    # A_mat = array([[4, -2, 1, 3],
    #                [1, 9, 2, 4],
    #                [4, 6, 1, 0],
    #                [6, 3, 9, 4]], float)
    # b_vec = array([2, 4, 5, 7], float)
    #
    # Gauss_Elimination(A_mat, b_vec)
    # Gauss_Elimination_Pivoting(A_mat, b_vec)


    # random_test_case test
    A_mat1, b_vec2 = random_test_case(200)
    # start timer
    t = time.time()
    Gauss_Elimination_Pivoting(A_mat1, b_vec2)
    elapsed_time = time.time() - t

    # times = []
    #
    # for i in range(10):
    #     t = time.time()
    #     A_mat1, b_vec2 = random_test_case(200)
    #     Gauss_Elimination_Pivoting(A_mat1, b_vec2)
    #     elapsed_time = time.time() - t
    #     times.append(elapsed_time)

    print("The elapsed time is: " + str(elapsed_time))


if __name__ == '__main__':
    main()
