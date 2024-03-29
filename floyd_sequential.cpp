#include <chrono>
#include <fstream>
#include <iostream>
#include <stdint.h>

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

    // Start Floyd algorithm
    auto start = std::chrono::steady_clock::now();
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                graph[i][j] = std::min(graph[i][j], graph[i][k] + graph[k][j]);
            }
        }
    }

    // Calculate total time and print it to console
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds> 
                            (std::chrono::steady_clock::now() - start);
    std::cout << "sequential: " << n << " " << duration.count() << std::endl;

    // Write Floyd result to file
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
