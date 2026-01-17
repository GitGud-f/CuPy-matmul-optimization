extern "C" {
    __global__ void matmul_kernel(const float* A, const float* B, float* C, int N) {
        

        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < N && col < N) {
            
            float sum = 0.0f;
            
            for (int k = 0; k < N; k++) {
                float a = A[row * N + k];
                float b = B[k * N + col];
                sum += a * b;
            }

            C[row * N + col] = sum;
        }
    }
}