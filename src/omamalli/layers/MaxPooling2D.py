import numpy as np
from .BaseLayer import BaseLayer

class MaxPooling2D(BaseLayer):
    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None

    def forward(self, input_array):
        self.input = input_array
        N, H, W, C = input_array.shape
        H_out = 1 + (H - self.pool_size) // self.stride
        W_out = 1 + (W - self.pool_size) // self.stride
        
        output = np.zeros((N, H_out, W_out, C))
        
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    for c in range(C):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        pool_region = input_array[n, h_start:h_end, w_start:w_end, c]
                        output[n, h, w, c] = np.max(pool_region)
        
        self.cache = (input_array, output)
        return output

    def backward(self, output_gradient, learning_rate):
        input_array, output = self.cache
        N, H, W, C = input_array.shape
        H_out, W_out, _ = output.shape[1:]
        input_gradient = np.zeros_like(input_array)
        
        for n in range(N):
            for h in range(H_out):
                for w in range(W_out):
                    for c in range(C):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        pool_region = input_array[n, h_start:h_end, w_start:w_end, c]
                        max_val = np.max(pool_region)
                        mask = pool_region == max_val
                        input_gradient[n, h_start:h_end, w_start:w_end, c] += mask * output_gradient[n, h, w, c]
        
        return input_gradient
