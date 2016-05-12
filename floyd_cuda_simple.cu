#include <chrono>
#include <cuda.h>
#include <fstream>
#include <iostream>

#include <stdint.h>
#include <stdio.h>

// Calculate one element of matrix per thread
__global__ void FloydSimple(uint32_t *graph, uint32_t *result, uint32_t n, int k) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n)
        return;
    result[i * n + j] = (graph[i * n + j] < (graph[i * n + k] + graph[k * n + j]))
                            ? graph[i * n + j] 
                            : (graph[i * n + k] + graph[k * n + j]);
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

    uint32_t *h_graph = new uint32_t[n * n];
    uint32_t *h_floyd_result = new uint32_t[n * n];
    for (size_t i = 0; i < n * n; ++i) {
        uint8_t current_elem;
        graph_reader.read((char *)&current_elem, 1);
        h_graph[i] = current_elem;
    }

    // Copy graph to device global memory
    auto start = std::chrono::steady_clock::now();

    uint32_t *d_graph, *d_floyd_result;
    cudaMalloc(&d_graph, sizeof(uint32_t) * n * n);
    cudaMalloc(&d_floyd_result, sizeof(uint32_t) * n * n);
    cudaMemcpy(d_floyd_result, h_graph, sizeof(uint32_t) * n * n, cudaMemcpyHostToDevice);
    cudaEvent_t iterationFinishedEvent;
    cudaEventCreate(&iterationFinishedEvent);

    // Start Floyd algorithm on cuda
    int threadNum = std::min(n, uint32_t(32));
    dim3 blockSize(threadNum, threadNum, 1);
    dim3 gridSize(n / threadNum + 1, n / threadNum + 1, 1);

    for (size_t k = 0; k < n; ++k) {
        std::swap(d_graph, d_floyd_result);
        // Start all threads for one iteration
        FloydSimple<<<gridSize, blockSize>>>(d_graph, d_floyd_result, n, k);
        cudaEventRecord(iterationFinishedEvent);
        cudaEventSynchronize(iterationFinishedEvent);
    }

    // Copy results to host
    cudaMemcpy(h_floyd_result, d_floyd_result, sizeof(int) * n * n, cudaMemcpyDeviceToHost);

    // Calculate all time used by cuda, and print it to console
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> 
                            (std::chrono::steady_clock::now() - start);
    std::cout << "time: " << duration.count() << std::endl;

    // Write Floyd results to file
    std::fstream result_writer(argv[2], std::fstream::out | std::fstream::binary);
    for (size_t i = 0; i < n * n; ++i) {
        result_writer.write((char*)&h_floyd_result[i], 4);
    }
    result_writer.close();

    delete[] h_graph;
    delete[] h_floyd_result;
    cudaFree(d_graph);
    cudaFree(d_floyd_result);
    return 0;
}
