import random
import time

# Create two 1000x1000 random matrices
def generate_matrix(size):
    return [[random.random() for _ in range(size)] for _ in range(size)]

size = 1000
A = generate_matrix(size)
B = generate_matrix(size)

# Matrix multiplication function
def matrix_multiply(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])

    C = [[0 for _ in range(p)] for _ in range(n)]

    for i in range(n):
        for j in range(p):
            sum = 0
            for k in range(m):
                sum += A[i][k] * B[k][j]
            C[i][j] = sum

    return C

# Outer loop to execute matrix multiplication 1000 times
start_time = time.time()

for _ in range(1000):
    result = matrix_multiply(A, B)

end_time = time.time()

# Measure the matrix multiplication computation time
total_time = end_time - start_time
print(f"Matrix multiplication computation time: {total_time:.2f} seconds")