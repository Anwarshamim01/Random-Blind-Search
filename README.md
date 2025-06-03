# Random-Blind-Search

This code was developed as part of the simulation work for my MS thesis. It implements random search (Lévy flight) processes on a 2D grid with periodic boundaries, tracking the time for a walker to return to its origin and to hit a new target. Both a Python (CPU) and a CUDA (GPU-accelerated) implementation are provided for performance comparison and large-scale simulation.

## Python Version
- File: `python code/blindsearchPythoncode.py`
- Requirements: Python 3, numpy, scipy, pandas, tqdm, matplotlib
- Output: Results are saved to `simulation_results.pkl` (pickle format)

### To run:
```sh
python "python code/blindsearchPythoncode.py"
```

## CUDA Version
- File: `cuda code/blindsearch.cu`
- Requirements: NVIDIA GPU, CUDA Toolkit, cuRAND library
- Output: Results are saved to `simulation_results.pickle` (binary format)

### To compile and run:
```sh
nvcc cuda code/blindsearch.cu -o simulation_executable -lcurand -lm
./simulation_executable
```

---
Both codes simulate multiple random walks, measure first return and first hit times, and save the results for further analysis.