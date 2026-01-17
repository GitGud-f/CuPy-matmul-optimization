extern "C" {

    __global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int N) {
        
        __shared__ float As[TILE_WIDTH][TILE_WIDTH];
        __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

        int bx = blockIdx.x;  int by = blockIdx.y;
        int tx = threadIdx.x; int ty = threadIdx.y;

        int row = by * TILE_WIDTH + ty;
        int col = bx * TILE_WIDTH + tx;

        float value = 0.0f;

        int num_phases = (N + TILE_WIDTH - 1) / TILE_WIDTH;

        for (int m = 0; m < num_phases; ++m) {


            
            int col_A = m * TILE_WIDTH + tx;
            if (row < N && col_A < N)
                As[ty][tx] = A[row * N + col_A];
            else
                As[ty][tx] = 0.0f;

            int row_B = m * TILE_WIDTH + ty;
            if (row_B < N && col < N)
                Bs[ty][tx] = B[row_B * N + col];
            else
                Bs[ty][tx] = 0.0f;

            __syncthreads();


            for (int k = 0; k < TILE_WIDTH; ++k) {
                value += As[ty][k] * Bs[k][tx];
            }

            __syncthreads();
        }

        if (row < N && col < N) {
            C[row * N + col] = value;
        }
    }
}