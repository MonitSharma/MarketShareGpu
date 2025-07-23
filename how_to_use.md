## Summary

This repository implements a GPU‑accelerated variant of **Schroeppel‑Shamir’s** algorithm for solving the market split problem and requires OpenMP (and optionally CUDA) to build and run locally. 

After cloning, you’ll install dependencies (libomp‑dev for OpenMP and CUDA Toolkit for GPU support), generate build files with CMake, compile with make, and then execute markshare_main with your chosen parameters. 

Below are step‑by‑step instructions to get the code up and running on Ubuntu.

# MarketShareGpu

## Description  
MarketShareGpu implements a GPU-accelerated variant of Schroeppel-Shamir’s algorithm for solving the market-share (market split) problem. It uses OpenMP for CPU-parallelism and optionally NVIDIA CUDA for GPU validation.

## Repository Layout  
- **CMakeLists.txt**  Top-level build configuration  
- **external/argparse/** Third-party command-line argument parser (git submodule)  
- **src/**       Source files (`.cpp`, `.cu`, headers)  
- **build/**     Out-of-source build directory (ignored by git)  
- **examples/**    (Optional) sample instance files  

## Prerequisites  
- **CMake**  ≥ 3.18  
- **GCC/Clang** with OpenMP support (`-fopenmp`)  
- **libomp-dev** (OpenMP headers/runtime)  
- **CUDA Toolkit** (optional, for `--gpu` mode)  

## Getting Started  

### 1. Clone the repository  
	git clone https://github.com/NCKempke/MarketShareGpu.git  
	cd MarketShareGpu  

### 2. Pull in submodules  
	git submodule update --init --recursive  

### 3. Create and enter build directory  
	mkdir build  
	cd build  

### 4. Configure the project  
	# CPU + CUDA (default)  
	cmake .. -DCMAKE_BUILD_TYPE=Release  

	# CPU-only (skip CUDA)  
	cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF  

### 5. Build  
	make -j$(nproc)  

## Usage  

### Show help  
	./markshare_main --help  

### Common flags  
- `-m`, `--m`  Rows (required unless using `-f`)  
- `-n`, `--n`  Columns (defaults to `(m-1)*10`)  
- `-k`, `--k`  Coefficient range `[0,k)` (default `100`)  
- `--reduce`  Max rows to reduce (requires reduction enabled)  
- `-s`, `--seed` Seed for random instance (default `2025`)  
- `-i`, `--iter` Number of instances to solve (default `1`)  
- `--gpu`   Run CUDA validation after CPU solve  
- `-f`, `--file` Load a saved instance (overrides `-m/-n/-k/-s/-i`)  
- `--max_pairs` Max pairs for GPU kernel (reduce if OOM)  

### Examples  

1. **CPU-only, default settings**  
	OMP_NUM_THREADS=32 ./markshare_main -m 7  

2. **Five random instances**  
	./markshare_main -m 8 -i 5  

3. **Custom range & seed**  
	./markshare_main -m 10 -k 500 -s 42  

4. **Enable GPU validation**  
	./markshare_main -m 9 --gpu  

5. **Load instance from file**  
	./markshare_main -f ../instances/inst.txt  

6. **Lower GPU memory usage**  
	./markshare_main -m 7 --gpu --max_pairs 1000000000  

## Troubleshooting  

- **Missing argparse submodule**  
  If CMake errors on `external/argparse`:  
	  git submodule update --init --recursive  

- **Unsupported GPU arch ‘compute_90’**  
  Remove `90` from `CUDA_ARCHITECTURES` in `CMakeLists.txt`, or upgrade CUDA ≥ 12.0.

- **OpenMP errors**  
  Ensure `libomp-dev` is installed and your compiler supports `-fopenmp`.

- **CUDA not found**  
  If you get “CUDA not found,” install the CUDA Toolkit and set:
	  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc  
	  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc  

## Contributing  
1. Fork the repo  
2. Create a feature branch  
3. Commit your changes & push  
4. Open a pull request  

## License  
This project is licensed under the MIT License.  
