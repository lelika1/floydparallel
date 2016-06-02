#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>

#include <stdint.h>
#include <stdio.h>

#define TILE_SIZE 32
#define STAGE_SIZE 16

#define HANDLE_ERROR(status) \
{ \
    if (status != cudaSuccess) \
    { \
        printf("%s failed  at line %d \nError message: %s \n", \
            __FILE__, __LINE__ ,cudaGetErrorString(status)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void WakeGpuKernel(int reps) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= reps) return;
}

__global__ void CalculateLeadBlock(uint32_t *graph, uint32_t n,
                                   uint32_t blockedIter)
{
    const int locI = threadIdx.y;
    const int locJ = threadIdx.x;

    const int glI = TILE_SIZE * blockedIter + locI;
    const int glJ = TILE_SIZE * blockedIter + locJ;
    if (glI >= n || glJ >= n || 
        glI >= TILE_SIZE * (blockedIter + 1) || glI < TILE_SIZE * blockedIter ||
        glJ >= TILE_SIZE * (blockedIter + 1) || glJ < TILE_SIZE * blockedIter)
    {
        return;
    }

    __shared__ uint32_t leadBlock[TILE_SIZE * TILE_SIZE];
    leadBlock[locI * TILE_SIZE + locJ] = graph[glI * n + glJ];
    __syncthreads();

    #pragma unroll
    for (size_t locIter = 0; locIter < TILE_SIZE; ++locIter) {
        uint32_t newPathLen = leadBlock[locI * TILE_SIZE + locIter]
                              + leadBlock[locIter * TILE_SIZE + locJ];
        if (newPathLen < leadBlock[locI * TILE_SIZE + locJ]) {
            leadBlock[locI * TILE_SIZE + locJ] = newPathLen;
        }
        __syncthreads();
    }
    graph[glI * n + glJ] = leadBlock[locI * TILE_SIZE + locJ];
}

__global__ void CalculateLeadRowAndColumn(uint32_t *graph, uint32_t n,
                                          uint32_t blockedIter)
{
    if (threadIdx.y * TILE_SIZE + threadIdx.x > TILE_SIZE * TILE_SIZE
        || blockIdx.x == blockedIter)
    {
        return;
    }

    int blockPosI, blockPosJ;
    if (blockIdx.y == 0) {
        // This is lead row
        blockPosI = blockedIter * TILE_SIZE;
        blockPosJ = blockIdx.x * TILE_SIZE;
    } else {
        // This is lead column
        blockPosI = blockIdx.x * TILE_SIZE;
        blockPosJ = blockedIter * TILE_SIZE;
    }

    int locI = threadIdx.y;
    int locJ = threadIdx.x;

    int glI = blockPosI + threadIdx.y;
    int glJ = blockPosJ + threadIdx.x;

    __shared__ uint32_t leadBlock[TILE_SIZE * STAGE_SIZE];
    __shared__ uint32_t curBlock[TILE_SIZE * TILE_SIZE];

    
    curBlock[locI * TILE_SIZE + locJ] = graph[glI * n + glJ];
    __syncthreads();

    size_t leadBlockOffset = blockedIter * TILE_SIZE;
    if (blockIdx.y == 0) {
        // This is lead row
        #pragma unroll
        for (size_t stage = 0; stage < TILE_SIZE / STAGE_SIZE; ++stage) {
            if (locI / STAGE_SIZE == stage) {
                leadBlock[locJ * STAGE_SIZE + (locI % STAGE_SIZE)] =
                    graph[(leadBlockOffset + locJ) * n + leadBlockOffset + locI];
            }
            __syncthreads();
            #pragma unroll
            for (size_t locIter = 0; locIter < STAGE_SIZE; ++locIter) {
                uint32_t newPathLen = curBlock[(stage * STAGE_SIZE + locIter) * TILE_SIZE + locJ]
                                      + leadBlock[locI * STAGE_SIZE + locIter];
                if (newPathLen < curBlock[locI * TILE_SIZE + locJ]) {
                    curBlock[locI * TILE_SIZE + locJ] = newPathLen;
                }
                __syncthreads();
            }
        }
    } else {
        // This is lead column
        #pragma unroll
        for (size_t stage = 0; stage < TILE_SIZE / STAGE_SIZE; ++stage) {
            if (locI / STAGE_SIZE == stage) {
                leadBlock[(locI % STAGE_SIZE) * TILE_SIZE + locJ] =
                    graph[(leadBlockOffset + locI) * n + leadBlockOffset + locJ];

            }
            __syncthreads();
            #pragma unroll
            for (size_t locIter = 0; locIter < STAGE_SIZE; ++locIter) {
                uint32_t newPathLen = curBlock[locI * TILE_SIZE + stage * STAGE_SIZE + locIter]
                                      + leadBlock[locIter * TILE_SIZE + locJ];
                if (newPathLen < curBlock[locI * TILE_SIZE + locJ]) {
                    curBlock[locI * TILE_SIZE + locJ] = newPathLen;
                }
                __syncthreads();
            }
        }
    }
    graph[glI * n + glJ] = curBlock[locI * TILE_SIZE + locJ];
}


__global__ void CalculateRestBlocks(uint32_t *graph, uint32_t n,
                                    uint32_t blockedIter)
{
    __shared__ uint32_t leadRow[TILE_SIZE * STAGE_SIZE];
    __shared__ uint32_t leadCol[TILE_SIZE * STAGE_SIZE];
    uint32_t curBlockElem;
    
    if (blockIdx.x == blockedIter
        || blockIdx.y == blockedIter)
    {
        return;
    }

    int blockPosI = blockIdx.y * TILE_SIZE;
    int blockPosJ = blockIdx.x * TILE_SIZE;

    int locI = threadIdx.y;
    int locJ = threadIdx.x;
    int glI = blockPosI + threadIdx.y;
    int glJ = blockPosJ + threadIdx.x;

    curBlockElem = graph[glI * n + glJ];
    __syncthreads();

    #pragma unroll
    for (int stage = 0; stage < TILE_SIZE / STAGE_SIZE; ++stage) {
        size_t leadBlocksOffset = blockedIter * TILE_SIZE;
        if (locI / STAGE_SIZE == stage) {
            leadRow[(locI % STAGE_SIZE)  * TILE_SIZE + locJ] = 
                graph[(leadBlocksOffset + locI) * n + (blockPosJ + locJ)];
            leadCol[locJ * STAGE_SIZE + (locI % STAGE_SIZE)] =
                graph[(blockPosI + locJ) * n + (leadBlocksOffset + locI)];
        }
        __syncthreads();
        #pragma unroll
        for (int locIter = 0; locIter < STAGE_SIZE; ++locIter) {
            uint32_t newPathLen = leadCol[locI * STAGE_SIZE + locIter]
                                  + leadRow[locIter * TILE_SIZE + locJ];
            if (newPathLen < curBlockElem) {
                curBlockElem = newPathLen;
            }
        }
        __syncthreads();
    }

    graph[glI * n + glJ] = curBlockElem;
}

__host__ void FloydBlocked(uint32_t *h_graph,
                           uint32_t *h_floydResult,
                           uint32_t n)
{
    // Copy graph to device global memory
    auto start = std::chrono::steady_clock::now();

    uint32_t *d_graph;
    cudaMalloc(&d_graph, sizeof(uint32_t) * n * n);
    cudaMemcpy(d_graph, h_graph, sizeof(uint32_t) * n * n, cudaMemcpyHostToDevice);

    dim3 firstStepGridSize(1, 1, 1);
    dim3 firstStepBlockSize(TILE_SIZE, TILE_SIZE, 1);

    dim3 secondStepGridSize((n - 1) / TILE_SIZE + 1, 2, 1);
    dim3 secondStepBlockSize(TILE_SIZE, TILE_SIZE, 1);

    dim3 thirdStepGridSize((n - 1)/ TILE_SIZE + 1,
                           (n - 1)/ TILE_SIZE + 1, 1);
    dim3 thirdStepBlockSize(TILE_SIZE, TILE_SIZE, 1);

    cudaError_t cudaStatus;
    cudaEvent_t stepFinishedEvent;
    cudaEventCreate(&stepFinishedEvent);
    for (int blockedIteration = 0; blockedIteration < n / TILE_SIZE; ++blockedIteration) {
        CalculateLeadBlock<<<firstStepGridSize, firstStepBlockSize>>>
                          (d_graph, n, blockedIteration);
        cudaStatus = cudaGetLastError();
        HANDLE_ERROR(cudaStatus);
        cudaEventRecord(stepFinishedEvent);
        cudaEventSynchronize(stepFinishedEvent);

        CalculateLeadRowAndColumn<<<secondStepGridSize, secondStepBlockSize>>>
                                 (d_graph, n, blockedIteration);
        cudaStatus = cudaGetLastError();
        HANDLE_ERROR(cudaStatus);
        cudaEventRecord(stepFinishedEvent);
        cudaEventSynchronize(stepFinishedEvent);

        CalculateRestBlocks<<<thirdStepGridSize, thirdStepBlockSize>>>
                           (d_graph, n, blockedIteration);
        cudaStatus = cudaGetLastError();
        HANDLE_ERROR(cudaStatus);
        cudaEventRecord(stepFinishedEvent);
        cudaEventSynchronize(stepFinishedEvent);
    }

    cudaStatus = cudaGetLastError();
    HANDLE_ERROR(cudaStatus);

    // Copy results to host
    cudaMemcpy(h_floydResult, d_graph, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    // Calculate all time used by cuda, and print it to console
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> 
                            (std::chrono::steady_clock::now() - start);
    std::cout << n << " " << duration.count() << std::endl;

    cudaFree(d_graph);
}


__host__ int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " graph_path results_path" << std::endl;
        return 1;
    }

    // Read vertex count and all graph
    uint32_t n;
    std::fstream graph_reader(argv[1], std::fstream::in | std::fstream::binary);
    graph_reader.read((char*)&n, 4);
    if (n % TILE_SIZE != 0) {
        std::cout << "Number of vertex shoud be divided by tile size (just for easier implementation). "
                  << "Tile size: " << TILE_SIZE << ". Vertex's count: " << n << "." 
                  << std::endl;
        graph_reader.close();
        return 1;
    }

    uint32_t *h_graph = new uint32_t[n * n];
    uint32_t *h_floydResult = new uint32_t[n * n];
    for (size_t i = 0; i < n * n; ++i) {
        uint8_t current_elem;
        graph_reader.read((char *)&current_elem, 1);
        h_graph[i] = current_elem;
    }
    graph_reader.close();

    // Run empty task on cuda - it will decrease time of first run
    int threadNum = std::min(n, uint32_t(32));
    dim3 gridSize(n / threadNum + 1, n / threadNum + 1, 1);
    dim3 cudaBlockSize(threadNum, threadNum, 1);
    WakeGpuKernel<<<1, cudaBlockSize>>>(32);

    // Blocked Floyd-Warshall algorithm on cuda
    FloydBlocked(h_graph, h_floydResult, n);

    // Write Floyd results to file
    std::fstream result_writer(argv[2], std::fstream::out | std::fstream::binary);
    for (size_t i = 0; i < n * n; ++i) {
        result_writer.write((char*)&h_floydResult[i], 4);
    }
    result_writer.close();

    delete[] h_graph;
    delete[] h_floydResult;

    return 0;
}
