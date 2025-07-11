# A GPU accelerated variant of Schroeppel-Shamir’s algorithm for solving the market split problem

The feasibility version of the market split problem (fMSP) as given in [[1]](#1) is equivalent to the n-dimensional subset sum problem: Find a vector $`x_j \in \{0,1\}^n`$ such that
```math
\begin{equation}
    \sum_{j=1}^n a_{ij} x_j = d_i \quad i = 1,\dots,m.
\end{equation}
```

Here, $`x_{j}`$ are binary decision variables, $`m, n \in \mathbb{N}`$, and we assume $`a_{ij}, d_i \in \mathbb{N}_0`$.

The algorithm implemented here can be found in [[2]](#2).

## Compile

This code requires OpenMP. On Ubuntu e.g. run
```
$ apt install libomp-dev
```
CUDA support is optional but recommended.

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE
make -j10
```

## Help
```
./markshare_main -h
Usage: markshare [--help] [--version] [--m VAR] [--n VAR] [--k VAR] [--reduce VAR] [--seed VAR] [--iter VAR] [--gpu] [--file VAR] [--max_pairs VAR]

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
  --max_pairs    Maximum number of pairs to be evaluated on the GPU simultaneously. If GPU runs OOM, reduce this number. [nargs=0..1] [default: 3500000000]
```

## Solving market split problems

```
OMP_NUM_THREADS=32 ./markshare_main -m 7 -k 100 -s 3 --reduce 1
```
Solves a market split problem (7,60) with coefficients in [0, 100) and random seed `3`. Not additional dimensionality reduction via surrogate constraint is applied.
Solving on CPU:
```
nkempke@opt-008541:~/scratch_htc/git/NCKempke/MarketShareGpu/build (main *)$ OMP_NUM_THREADS=32 ./markshare_main -m 7 -k 100 -s 3 --reduce 1   
Storing mark share instance as markshare_m_7_n_60_seed_3.prb
Running markshare: m=7, n=60, seed=3, iter=0, nthread=32
[
 [ 55  7 70 83 29 12 51 56 89 43 89  1 12  4 20 24  5  9 44 69  2 14 45 45 64 21 27 35 67 49 59 91  2 76 55 97 25 40 41 55 28 30 69 58 44 27 15 45 54 75 78 81 30 25 22 89 38 71 93  2 | 1328 ]
 [ 97 14 67 32 90 80 84 39 37 42  9 41 65 78 55 71 36  2 22 31 40 56 46 74 26 60 29 20 45 65 86 66 58 87 28 81 27 37 45  7 20 76 20  6 51 84  8 69 48 44 36 36 70 51 74 45 69 80 68 31 | 1480 ]
 [ 37 84 66 48 33 54 57 82 32 52 44 14  6 30 24 80 97 31 23  2 69 48 65  2 72 50 47 91 59 20  6 92  7 27 19 17 15 99 10 69 12 75 55 11 18  8 95 66 68 32 54 36 70 52 26 42 92 81 83 57 | 1406 ]
 [ 72 88 48 80 84 34 74 77 66 89 91 78 63 28 36 88 55 54 19 82 19 23 72 20 78 95 97 76 85 18 54 46  8 50 48 17 92 29 78 26 48 88 45 50 21 57 17 34  7 60 89 12 64  3 14 99 41 26  4 25 | 1570 ]
 [ 20 27 73 25 65 89 47 83 27 14 65 99 95 35 43 54  7 49  5 12  8 86 95 21 54 82 83 99 17 99 26 15 69 47 89 70 34 34  6 69 86 74 29 81 74 84 15  8 69 30 84 56 72 91 35  9 72 40 13 53 | 1556 ]
 [ 31 64 41 30 87 84 15 49 88 70 79 67 97 62 36 99 20 15 24 81 82 30 96 10 69 28 48 71 28 23 83 69 87  2  9 92 21 36 83 80 84 71 31 60 27 23 43 44 53 64  9 78 83 79 53 38 77 40 23 10 | 1588 ]
 [ 96 43 75 85 34 96 94 72 70 41 84 13  4  2  5 22 74 35 30 78 51 44 15 47 97 51 50 36 82 26  7 80 47 90  6 36 88 88 44 30  6 28  7 21 53 42  7 12 18 99 43 43 49 61 58 97 62 29 37 45 | 1442 ]
]
Running reduced dim shroeppel shamir
Running with 32 threads
Max reducible dimension is 4 (encoded with basis 12001)
Reducing 1 dimensions for Shroeppel-Shamir - leaving 6 for verification
Splitting sets into [0, 14]; [15, 29]; [30, 44]; [45, 60]
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Running the search loop

    1     0.04s:      0 +   1328; 1 x 1728 possible solutions
    2     0.04s:      1 +   1327; 1 x 1844 possible solutions
    3     0.04s:      2 +   1326; 1 x 1857 possible solutions
    4     0.04s:      3 +   1325; 1 x 1961 possible solutions
    5     0.04s:      4 +   1324; 1 x 2079 possible solutions
    6     0.04s:      5 +   1323; 2 x 2090 possible solutions
    7     0.04s:      6 +   1322; 2 x 2269 possible solutions
    8     0.05s:      7 +   1321; 3 x 2256 possible solutions
    9     0.05s:      8 +   1320; 2 x 2369 possible solutions
   10     0.05s:      9 +   1319; 3 x 2428 possible solutions
   20     0.06s:     19 +   1309; 11 x 3244 possible solutions
   30     0.07s:     29 +   1299; 22 x 4231 possible solutions
   40     0.08s:     39 +   1289; 48 x 5572 possible solutions
   50     0.09s:     49 +   1279; 88 x 7332 possible solutions
   60     0.10s:     59 +   1269; 166 x 9464 possible solutions
   70     0.11s:     69 +   1259; 294 x 12003 possible solutions
   80     0.12s:     79 +   1249; 485 x 15323 possible solutions
   90     0.13s:     89 +   1239; 794 x 19425 possible solutions
  100     0.15s:     99 +   1229; 1236 x 24240 possible solutions
  200     0.54s:    199 +   1129; 39417 x 162375 possible solutions
  300     4.28s:    299 +   1029; 354871 x 626117 possible solutions
Found market share solution from SS-Algorithm!
Subset for index 6916 (binary 0000000000000000000000000000000000000000000000000001101100000100): 70, 89, 43, 1, 12
Subset for index 19506 (binary 0000000000000000000000000000000000000000000000000100110000110010): 5, 69, 2, 21, 27, 49
Subset for index 11227 (binary 0000000000000000000000000000000000000000000000000010101111011011): 59, 91, 76, 55, 25, 40, 41, 55, 30, 58
Subset for index 410 (binary 0000000000000000000000000000000000000000000000000000000110011010): 15, 54, 75, 30, 25
The sum is 215 + 173 + 530 + 199 = 1117
Writing solution to markshare_m_7_n_60_seed_3.sol
Found feasible solution!
Evaluate solutions CPU      	 total time    31.81s
Eval CPU: Hash table search 	 total time     2.52s
Eval CPU: Hash table setup  	 total time    18.10s
Candidate extraction        	 total time     0.75s
List traversal              	 total time    32.07s
Solution time               	 total time    32.10s
Setup time                  	 total time     0.04s
```
The same instance on GPU:
```
OMP_NUM_THREADS=32 ./markshare_main -m 7 -k 100 -s 3 --reduce 1 --gpu
Storing mark share instance as markshare_m_7_n_60_seed_3.prb
Running markshare: m=7, n=60, seed=3, iter=0, nthread=32
[
 [ 55  7 70 83 29 12 51 56 89 43 89  1 12  4 20 24  5  9 44 69  2 14 45 45 64 21 27 35 67 49 59 91  2 76 55 97 25 40 41 55 28 30 69 58 44 27 15 45 54 75 78 81 30 25 22 89 38 71 93  2 | 1328 ]
 [ 97 14 67 32 90 80 84 39 37 42  9 41 65 78 55 71 36  2 22 31 40 56 46 74 26 60 29 20 45 65 86 66 58 87 28 81 27 37 45  7 20 76 20  6 51 84  8 69 48 44 36 36 70 51 74 45 69 80 68 31 | 1480 ]
 [ 37 84 66 48 33 54 57 82 32 52 44 14  6 30 24 80 97 31 23  2 69 48 65  2 72 50 47 91 59 20  6 92  7 27 19 17 15 99 10 69 12 75 55 11 18  8 95 66 68 32 54 36 70 52 26 42 92 81 83 57 | 1406 ]
 [ 72 88 48 80 84 34 74 77 66 89 91 78 63 28 36 88 55 54 19 82 19 23 72 20 78 95 97 76 85 18 54 46  8 50 48 17 92 29 78 26 48 88 45 50 21 57 17 34  7 60 89 12 64  3 14 99 41 26  4 25 | 1570 ]
 [ 20 27 73 25 65 89 47 83 27 14 65 99 95 35 43 54  7 49  5 12  8 86 95 21 54 82 83 99 17 99 26 15 69 47 89 70 34 34  6 69 86 74 29 81 74 84 15  8 69 30 84 56 72 91 35  9 72 40 13 53 | 1556 ]
 [ 31 64 41 30 87 84 15 49 88 70 79 67 97 62 36 99 20 15 24 81 82 30 96 10 69 28 48 71 28 23 83 69 87  2  9 92 21 36 83 80 84 71 31 60 27 23 43 44 53 64  9 78 83 79 53 38 77 40 23 10 | 1588 ]
 [ 96 43 75 85 34 96 94 72 70 41 84 13  4  2  5 22 74 35 30 78 51 44 15 47 97 51 50 36 82 26  7 80 47 90  6 36 88 88 44 30  6 28  7 21 53 42  7 12 18 99 43 43 49 61 58 97 62 29 37 45 | 1442 ]
]
Running reduced dim shroeppel shamir
Running with 32 threads
Max reducible dimension is 4 (encoded with basis 12001)
Reducing 1 dimensions for Shroeppel-Shamir - leaving 6 for verification
Splitting sets into [0, 14]; [15, 29]; [30, 44]; [45, 60]
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Running the search loop

    1     0.10s [0.006295 GB]:      0 +   1328; 1 x 1728 possible solutions
    2     0.10s [0.006295 GB]:      1 +   1327; 1 x 1844 possible solutions
    3     0.11s [0.006333 GB]:      2 +   1326; 1 x 1857 possible solutions
    4     0.11s [0.006333 GB]:      3 +   1325; 1 x 1961 possible solutions
    5     0.11s [0.006333 GB]:      4 +   1324; 1 x 2079 possible solutions
    6     0.11s [0.006335 GB]:      5 +   1323; 2 x 2090 possible solutions
    7     0.11s [0.006338 GB]:      6 +   1322; 2 x 2269 possible solutions
    8     0.12s [0.006338 GB]:      7 +   1321; 3 x 2256 possible solutions
    9     0.12s [0.006341 GB]:      8 +   1320; 2 x 2369 possible solutions
   10     0.12s [0.006344 GB]:      9 +   1319; 3 x 2428 possible solutions
   20     0.12s [0.006354 GB]:     19 +   1309; 11 x 3244 possible solutions
   30     0.12s [0.006372 GB]:     29 +   1299; 22 x 4231 possible solutions
   40     0.12s [0.006396 GB]:     39 +   1289; 48 x 5572 possible solutions
   50     0.12s [0.006423 GB]:     49 +   1279; 88 x 7332 possible solutions
   60     0.12s [0.006458 GB]:     59 +   1269; 166 x 9464 possible solutions
   70     0.13s [0.006487 GB]:     69 +   1259; 294 x 12003 possible solutions
   80     0.13s [0.006535 GB]:     79 +   1249; 485 x 15323 possible solutions
   90     0.13s [0.006594 GB]:     89 +   1239; 794 x 19425 possible solutions
  100     0.14s [0.006655 GB]:     99 +   1229; 1236 x 24240 possible solutions
  200     0.21s [0.008643 GB]:    199 +   1129; 39417 x 162375 possible solutions
  300     0.39s [0.016122 GB]:    299 +   1029; 354871 x 626117 possible solutions
Found market share solution from SS-Algorithm!
Subset for index 6916 (binary 0000000000000000000000000000000000000000000000000001101100000100): 70, 89, 43, 1, 12
Subset for index 19506 (binary 0000000000000000000000000000000000000000000000000100110000110010): 5, 69, 2, 21, 27, 49
Subset for index 11227 (binary 0000000000000000000000000000000000000000000000000010101111011011): 59, 91, 76, 55, 25, 40, 41, 55, 30, 58
Subset for index 410 (binary 0000000000000000000000000000000000000000000000000000000110011010): 15, 54, 75, 30, 25
The sum is 215 + 173 + 530 + 199 = 1117
Writing solution to markshare_m_7_n_60_seed_3.sol
Found feasible solution!
Eval GPU: check results     	 total time     0.02s
Eval GPU: binary search     	 total time     0.26s
Eval GPU: sort required     	 total time     0.16s
Candidate extraction        	 total time     0.53s
List traversal              	 total time     0.63s
Solution time               	 total time     0.72s
Evaluate solutions GPU      	 total time     0.49s
Setup time                  	 total time     0.10s
```
On GPU, an approximate amount of GPU memory consumed is reported in each info line. If the GPU runs OOM (note that this happens before the memory consumed actually reaches the GPU memory size), try to reduce `--max-pairs`.

Running `--gpu` without CUDA support will result in an abort:
```
Storing mark share instance as markshare_m_7_n_60_seed_3.prb
Running markshare: m=7, n=60, seed=3, iter=0, nthread=32
[
 [ 55  7 70 83 29 12 51 56 89 43 89  1 12  4 20 24  5  9 44 69  2 14 45 45 64 21 27 35 67 49 59 91  2 76 55 97 25 40 41 55 28 30 69 58 44 27 15 45 54 75 78 81 30 25 22 89 38 71 93  2 | 1328 ]
 [ 97 14 67 32 90 80 84 39 37 42  9 41 65 78 55 71 36  2 22 31 40 56 46 74 26 60 29 20 45 65 86 66 58 87 28 81 27 37 45  7 20 76 20  6 51 84  8 69 48 44 36 36 70 51 74 45 69 80 68 31 | 1480 ]
 [ 37 84 66 48 33 54 57 82 32 52 44 14  6 30 24 80 97 31 23  2 69 48 65  2 72 50 47 91 59 20  6 92  7 27 19 17 15 99 10 69 12 75 55 11 18  8 95 66 68 32 54 36 70 52 26 42 92 81 83 57 | 1406 ]
 [ 72 88 48 80 84 34 74 77 66 89 91 78 63 28 36 88 55 54 19 82 19 23 72 20 78 95 97 76 85 18 54 46  8 50 48 17 92 29 78 26 48 88 45 50 21 57 17 34  7 60 89 12 64  3 14 99 41 26  4 25 | 1570 ]
 [ 20 27 73 25 65 89 47 83 27 14 65 99 95 35 43 54  7 49  5 12  8 86 95 21 54 82 83 99 17 99 26 15 69 47 89 70 34 34  6 69 86 74 29 81 74 84 15  8 69 30 84 56 72 91 35  9 72 40 13 53 | 1556 ]
 [ 31 64 41 30 87 84 15 49 88 70 79 67 97 62 36 99 20 15 24 81 82 30 96 10 69 28 48 71 28 23 83 69 87  2  9 92 21 36 83 80 84 71 31 60 27 23 43 44 53 64  9 78 83 79 53 38 77 40 23 10 | 1588 ]
 [ 96 43 75 85 34 96 94 72 70 41 84 13  4  2  5 22 74 35 30 78 51 44 15 47 97 51 50 36 82 26  7 80 47 90  6 36 88 88 44 30  6 28  7 21 53 42  7 12 18 99 43 43 49 61 58 97 62 29 37 45 | 1442 ]
]
Running reduced dim shroeppel shamir
Running with 32 threads
Max reducible dimension is 4 (encoded with basis 12001)
Reducing 1 dimensions for Shroeppel-Shamir - leaving 6 for verification
Splitting sets into [0, 14]; [15, 29]; [30, 44]; [45, 60]
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Generating 32768 possible subsets for as set of size 15.
Running the search loop

    1     0.04s:      0 +   1328; 1 x 1728 possible solutions
Error: GPU mode not available!

Aborting!
```

## References

<a id="1">[1]</a> 
Cornéujols, G., & Dawande, M. (1998).
A Class of Hard Small 0-1 Programs.
In *Lecture Notes in Computer Science* (pp. 284-293). Springer Berlin Heidelberg.
https://doi.org/10.1007/3-540-69346-7\_22

<a id="2">[2]</a> 
Kempke, N.-C., & Koch, T. (2025).
A GPU accelerated variant of Schroeppel-Shamir’s algorithm for solving the market split problem.
TODO JOURNAL
ArxivLINK!
