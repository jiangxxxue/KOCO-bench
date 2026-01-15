#include <iostream>
#include <fstream>
#include <vector>

int main() {
    // Corrected line: Use double quotes for strings in C++
    const char* bin_file = "/home/fontan/VSLAM-LAB/VSLAM-Baselines/r2d2/info.bin";

    // Open the file in binary mode
    std::ifstream infile(bin_file, std::ios::binary);

    if (!infile) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    // Find the size of the file
    infile.seekg(0, std::ios::end);
    std::streamsize file_size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    // Assuming the data type is float (4 bytes per value)
    std::size_t num_elements = file_size / sizeof(float);

    // Allocate memory for the data
    std::vector<float> data(num_elements);

    // Read the binary data into the vector
    if (infile.read(reinterpret_cast<char*>(data.data()), file_size)) {
        std::cout << "Successfully read " << num_elements << " elements." << std::endl;
    } else {
        std::cerr << "Error reading file!" << std::endl;
    }

    infile.close();

    // Now, you can use the 'data' vector to access the values
    std::cout << "num_elements ========================================== " << num_elements << std::endl;
    for (std::size_t i = 0; i < num_elements; ++i) {
        std::cout << "Element " << i << ": " << data[i] << std::endl;
    }

    return 0;
}
