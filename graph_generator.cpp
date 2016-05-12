#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cout << "usage: " << argv[0] << " n graph_path" << std::endl;
        return 1;
    }
    uint32_t n = atoi(argv[1]);
    std::fstream out(argv[2], std::fstream::out | std::fstream::binary);
    out.write((char*)&n, 4);

    uint8_t current_element;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j) {
                current_element = 0;
            } else {
                current_element = rand() % 255;
            }
            out.write((char*)&current_element, 1);
        }
    }
    out.close();
    return 0;
}
