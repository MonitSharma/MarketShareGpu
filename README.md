A GPU accelerated variant of Schroeppel-Shamir’s algorithm for solving the market split problem

The algorithm implemented here can be found in TODO.

This code requires OpenMP. On Ubunutu e.g. run
```
$ apt install libomp-dev
```
CUDA support is optional but recommended.

To compile it:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j10
```

To get help on available parameters
```
./markshare_main -h
Usage: markshare [--help] [--version] [--m VAR] [--n VAR] [--k VAR] [--reduce VAR] [--seed VAR] [--iter VAR] [--gpu] [--file VAR]

Optional arguments:
  -h, --help     shows help message and exits 
  -v, --version  prints version information and exits 
  -m, --m        Number of rows of the markshare problem. 
  -n, --n        Number of columns of the markshare problem. Set to (m - 1) * 10 if not given.  
  -k, --k        Coefficients are generated in the range [0, k). [nargs=0..1] [default: 100]
  --reduce       Number of rows (max) to be reduced. Only effective if --reduced is set.  [nargs=0..1] [default: 0]
  -s, --seed     Random seed for instance generation. [nargs=0..1] [default: 2025]
  -i, --iter     Number of problems to solve. Seed for problem of iteration i (starting from 0) is seed + i. [nargs=0..1] [default: 1]
  --gpu          Run validation on GPU 
  -f, --file     Supply instance path to read instance from. Overrides '-m', '-n', '-k', and '-i'
```
Solving market split problems (special n-dimensional subset sum problems):
```
OMP_NUM_THREADS=32 ./markshare_main -m 7 -k 100 -s 3 --reduce 1
```
Solves a market split problem (7,60) with coefficients in [0, 100) and random seed `3`. Not additional dimensionality reduction via surrogate constraint is applied.
