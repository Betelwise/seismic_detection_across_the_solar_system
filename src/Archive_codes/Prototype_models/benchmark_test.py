# import tensorflow as tf
# import numpy as np
# import time

# # Generate random data
# x = np.random.rand(10000, 1000).astype(np.float32)

# # Define a simple operation
# def matmul_on_gpu():
#     with tf.device('/GPU:0'):
#         a = tf.constant(x)
#         b = tf.constant(np.random.rand(1000, 2000).astype(np.float32))  # Adjusted shape
#         c = tf.matmul(a, b)  # Now, a is (10000, 1000) and b is (1000, 2000)

#     return c

# # Measure execution time
# start_time = time.time()
# result = matmul_on_gpu()
# end_time = time.time()

# print(f"Time taken for matrix multiplication on GPU: {end_time - start_time:.4f} seconds")
import tensorflow as tf
import numpy as np
import time

# Generate larger random data
x = np.random.rand(10000, 10000).astype(np.float32)  # Increase size to 10000 x 10000

def matmul_on_cpu():
    with tf.device('/CPU:0'):
        a = tf.constant(x)
        b = tf.constant(np.random.rand(10000, 10000).astype(np.float32))  # Match dimensions for multiplication
        c = tf.matmul(a, b)
    return c

def matmul_on_gpu():
    with tf.device('/GPU:0'):
        a = tf.constant(x)
        b = tf.constant(np.random.rand(10000, 10000).astype(np.float32))  # Match dimensions for multiplication
        c = tf.matmul(a, b)
    return c

# Measure execution time for CPU
start_time_cpu = time.time()
result_cpu = matmul_on_cpu()
end_time_cpu = time.time()

# Measure execution time for GPU
start_time_gpu = time.time()
result_gpu = matmul_on_gpu()
end_time_gpu = time.time()

print(f"Time taken for matrix multiplication on CPU: {end_time_cpu - start_time_cpu:.4f} seconds")
print(f"Time taken for matrix multiplication on GPU: {end_time_gpu - start_time_gpu:.4f} seconds")
