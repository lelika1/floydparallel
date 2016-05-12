#include <chrono>

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <omp.h>


int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " graph_path results_path" << std::endl;
        return 1;
    }

    // Read vertex count and all graph
    uint32_t n;
    std::fstream graph_reader(argv[1], std::fstream::in | std::fstream::binary);
    graph_reader.read((char*)&n, 4);

    uint32_t **graph = new uint32_t*[n];
    for (size_t i = 0; i < n; ++i) {
        graph[i] = new uint32_t[n];
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uint8_t current_elem;
            graph_reader.read((char *)&current_elem, 1);
            graph[i][j] = current_elem;
        }
    }
    graph_reader.close();

    // Start parallel Floyd
    auto start = std::chrono::steady_clock::now();
    for (size_t k = 0; k < n; ++k) {
        size_t i, j;
        #pragma omp parallel for private(i,j)
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                graph[i][j] = std::min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }

    // Calculate total duration and print to console
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> 
                            (std::chrono::steady_clock::now() - start);
    std::cout << "open_mp: " << n << " " << duration.count() << std::endl;

    // Write results to file
    std::fstream result_writer(argv[2], std::fstream::out | std::fstream::binary);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result_writer.write((char *)&graph[i][j], 4);
        }
    }
    result_writer.close();

    delete[] graph;
    return 0;
}
