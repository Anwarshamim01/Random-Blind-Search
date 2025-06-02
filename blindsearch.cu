#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <curand.h>         // For host-side cuRAND functions
#include <curand_kernel.h>  // For device-side cuRAND functions and state types
#include <float.h>          // For FLT_EPSILON or DBL_EPSILON

// Helper function for checking CUDA Runtime API errors
#define CHECK_CUDA_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
struct RunData {
    int run_index;
    int first_return_time;
    int first_hit_time;
    // Note: `visited_targets` is not included here due to complexity.
};
// Helper function for checking cuRAND API errors
#define CHECK_CURAND_ERROR(ans) { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char *file, int line, bool abort=true)
{
   if (code != CURAND_STATUS_SUCCESS) {
      fprintf(stderr,"cuRANDAssert: %d (Error code) in %s %d\n", code, file, line); // cuRAND doesn't have string messages like CUDA
      if (abort) exit(code);
   }
}


// Periodic boundary function for device
__device__ inline float periodic_boundaries_d(float pos, float boundary) {
    // Equivalent to Python's ((pos + boundary) % (2 * boundary)) - boundary
    // Handles potential negative results from fmodf
    float result = fmodf(pos + boundary, 2.0f * boundary);
    if (result < 0) {
        result += 2.0f * boundary;
    }
    return result - boundary;
}

// Alpha-stable Levy step generation on device (Zolatorev/Mantegna-type method)
// Using double precision for intermediate calculations for better accuracy
// Takes the specific Philox state type
__device__ inline double levy_alpha_stable_d(curandStatePhilox4_32_10_t* rng_state, double alpha, double beta, double mu, double c, double delta_t) {
    // theta ~ U(-pi/2, pi/2)
    double theta = curand_uniform_double(rng_state) * M_PI - M_PI_2;

    // W ~ Exp(1) generated from Uniform using inverse transform: W = -log(U)
    double U_exp = curand_uniform_double(rng_state);
    // Ensure U_exp > 0 to avoid log(0)
    U_exp = fmax(U_exp, DBL_EPSILON); // Using fmax with double precision
    double W = -log(U_exp);

    // Ensure W is not zero to avoid division by zero
    W = fmax(W, 1e-10); // Using fmax with double precision


    double num = sin(alpha * theta);
    double den = pow(cos(theta), 1.0 / alpha);
    double cos_term = cos((1.0 - alpha) * theta);

    double correction_factor;
    // Handle potential negative base for pow, though for alpha in (0, 2), this shouldn't be an issue
    // for cos_term >= 0. If cos_term < 0, the Levy formula derivation typically implies alpha=1 or specific cases.
    // Assuming the Python logic handles this correctly by using abs implicitly or the formula structure prevents it.
    // Let's follow the Python's likely intent for this specific formula structure.
    // Ensure the base of pow is non-negative
    double pow_base = fabs(cos_term) / W;
    if (pow_base < 0) pow_base = 0; // Should not happen with fabs, but safeguard

    double exponent = (1.0 - alpha) / alpha;

    correction_factor = pow(pow_base, exponent);


    double step_length = (num / den) * correction_factor;

    if (isnan(step_length) || isinf(step_length)) {
        step_length = 0.0;
    }

    return c * step_length + mu * delta_t;
}

// CUDA Kernel for simulating Levy flights
__global__ void simulate_levy_runs_kernel(
    int num_runs,
    int max_time,
    float alpha, float beta, float mu, float c, float delta_t,
    float vision_radius_sq, // Squared vision radius for faster comparison
    float boundary,
    const float* d_targets, // Device pointer to target positions (x, y pairs)
    int num_targets,
    int starting_target_index, // Index of the starting target
    const float* d_starting_pos, // Device pointer to starting position (x, y)
    int* d_first_return_time,   // Device pointer for output (initialized to -1)
    int* d_first_hit_time,      // Device pointer for output (initialized to -1)
    curandStatePhilox4_32_10_t* rng_states  // Device pointer for RNG states (using specific type)
) {
    int run_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (run_idx >= num_runs) {
        return;
    }

    // Load random number generator state for this thread
    curandStatePhilox4_32_10_t rng_state = rng_states[run_idx];

    float walker_pos_x = d_starting_pos[0];
    float walker_pos_y = d_starting_pos[1];

    int step_count = 0;
    int first_return_time = -1;
    int first_hit_time = -1;

    // Simulation loop
    while (step_count < max_time) { // Continue as long as max_time not reached

        // Generate Levy step using double precision internally
        double jump_length_double = fabs(levy_alpha_stable_d(&rng_state, alpha, beta, mu, c, delta_t));
        double angle = curand_uniform_double(&rng_state) * 2.0 * M_PI;

        double dx_total_double = jump_length_double * cos(angle);
        double dy_total_double = jump_length_double * sin(angle);

        // Determine number of small increments
        // The Python code has a loop to ensure increments are <= 1.0
        // Lets replicate that logic, ensuring at least one increment
        double dd_double = sqrt(dx_total_double*dx_total_double + dy_total_double*dy_total_double);

        // Replace fmax(1, (int)dd_double) with a manual check
        int num_increments = (int)dd_double;
        if (num_increments < 1) {
            num_increments = 1;
        }

        // Adjust num_increments until step size per increment is <= 1.0
        // Avoid infinite loops if dd_double is huge (e.g., from an extreme Levy step)
        if (dd_double > 1e15) { // Add a safeguard for extremely large jumps
            num_increments = (int)dd_double; // Just use dd as num_increments in extreme cases
            if (num_increments <= 0) num_increments = 1; // Ensure at least 1
        } else {
             // Only perform the while loop adjustment if not an extremely large jump
             // The initial check `if (num_increments < 1) num_increments = 1;` handles dd_double <= 1.0 cases
             while (num_increments > 0 && dd_double / num_increments > 1.0) {
                 num_increments++;
            }
            // Re-check after while loop just in case num_increments somehow became 0
            if (num_increments == 0) num_increments = 1;
        }

        // Handle case where dd_double is 0 (no jump)
        if (dd_double == 0) num_increments = 1;


        float dx_step = dx_total_double / num_increments;
        float dy_step = dy_total_double / num_increments;

        // Loop over increments
        for (int i = 0; i < num_increments; ++i) {
            if (step_count >= max_time) {
                break; // Break inner loop if max_time is reached during increments
            }

            walker_pos_x += dx_step;
            walker_pos_y += dy_step;

            // Apply periodic boundaries
            walker_pos_x = periodic_boundaries_d(walker_pos_x, boundary);
            walker_pos_y = periodic_boundaries_d(walker_pos_y, boundary);

            step_count++;

            // Check for target detection (linear scan for nearest target)
            // This is the most performance-critical part and could be optimized
            float min_dist_sq = 1e18; // Initialize with a large value
            int nearest_target_idx = -1;

            for (int j = 0; j < num_targets; ++j) {
                float target_x = d_targets[j * 2];
                float target_y = d_targets[j * 2 + 1];
                float dist_sq = (walker_pos_x - target_x) * (walker_pos_x - target_x) +
                                (walker_pos_y - target_y) * (walker_pos_y - target_y);

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    nearest_target_idx = j;
                }
            }

            // Check if detected and record first times
            if (nearest_target_idx != -1 && min_dist_sq <= vision_radius_sq) {
                if (nearest_target_idx == starting_target_index && step_count > 60 && first_return_time == -1) {
                    first_return_time = step_count;
                } else if (nearest_target_idx != starting_target_index && first_hit_time == -1) {
                    first_hit_time = step_count;
                }
            }

             // Stop simulation for this run if both events have occurred
            if (first_return_time != -1 && first_hit_time != -1) {
                break; // Break inner loop
            }
        } // End of inner loop (increments)

        // Break outer loop if both events have occurred or max_time is reached
        if (first_return_time != -1 && first_hit_time != -1) {
            break; // Break outer loop
        }
         if (step_count >= max_time) {
            break; // Break outer loop if max_time reached
        }
    } // End of simulation loop

    // Store results in global memory
    d_first_return_time[run_idx] = first_return_time;
    d_first_hit_time[run_idx] = first_hit_time;

    // Save the updated RNG state (optional for independent runs)
    // rng_states[run_idx] = rng_state; // Not strictly needed if state is init per kernel launch
}

// Kernel to initialize device RNG states
__global__ void init_rng_states_kernel(curandStatePhilox4_32_10_t* states, unsigned long long base_seed, int num_threads) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_threads) {
        // Use thread index as sequence, base_seed as initial seed
        curand_init(base_seed, idx, 0, &states[idx]);
    }
}


int main() {
    // Parameters (matching Python code)
    float alpha = 0.6f;
    float beta = 0.0f;
    float mu = 0.0f;
    float c = 1.0f;
    float delta_t = 1.0f; // Not directly used in step calculation magnitude in Python, but kept for structure
    int max_time = 100000;
    float vision_radius = 3.0f;
    float vision_radius_sq = vision_radius * vision_radius; // Use squared radius
    int num_runs =200000;
    float boundary = 12000.0f;
    int num_targets = 1200;

    // --- Target setup ---
    int grid_size = static_cast<int>(sqrt(num_targets));
    // Ensure num_targets is a perfect square or handle grid size appropriately
    if (grid_size * grid_size != num_targets) {
         fprintf(stderr, "Warning: num_targets is not a perfect square. Adjusting grid_size.\n");
         // Simple approximation, might not perfectly match Python's grid
         grid_size = static_cast<int>(sqrt(num_targets));
         // To get exactly num_targets points, a different strategy might be needed
         // For this conversion, assume grid_size * grid_size is close to num_targets
         // and generate grid_size*grid_size points.
         num_targets = grid_size * grid_size;
         fprintf(stderr, "Adjusted num_targets to %d based on grid_size %d.\n", num_targets, grid_size);
    }

    float* targets = (float*)malloc(num_targets * 2 * sizeof(float)); // x, y pairs
    if (targets == NULL) {
        fprintf(stderr, "Failed to allocate host memory for targets.\n");
        return 1;
    }

    float x_coords[grid_size];
    float y_coords[grid_size];

    for (int i = 0; i < grid_size; ++i) {
        x_coords[i] = -boundary + (2.0f * boundary / (grid_size - 1)) * i;
        y_coords[i] = -boundary + (2.0f * boundary / (grid_size - 1)) * i;
    }

    int target_idx = 0;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            targets[target_idx * 2] = x_coords[i];
            targets[target_idx * 2 + 1] = y_coords[j];
            target_idx++;
        }
    }

    // --- Starting position ---
    int middle_index = (grid_size / 2) * grid_size + (grid_size / 2);
    float starting_pos[2];
    starting_pos[0] = targets[middle_index * 2];
    starting_pos[1] = targets[middle_index * 2 + 1];

    // --- CUDA Setup ---
    float* d_targets;
    float* d_starting_pos;
    int* d_first_return_time;
    int* d_first_hit_time;
    curandStatePhilox4_32_10_t* rng_states; // Use specific state type

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_targets, num_targets * 2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_starting_pos, 2 * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_first_return_time, num_runs * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_first_hit_time, num_runs * sizeof(int)));
    // Allocate memory for specific cuRAND states
    CHECK_CUDA_ERROR(cudaMalloc(&rng_states, num_runs * sizeof(curandStatePhilox4_32_10_t)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_targets, targets, num_targets * 2 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_starting_pos, starting_pos, 2 * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize output arrays on device to -1 (or another sentinel value)
    CHECK_CUDA_ERROR(cudaMemset(d_first_return_time, -1, num_runs * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_first_hit_time, -1, num_runs * sizeof(int)));

    // --- Initialize Device RNG States Kernel ---
    // This kernel initializes each thread's RNG state
    // using its thread index as the sequence number.
    unsigned long long seed = 12345ULL; // Base seed for reproducibility

    int threads_per_block_init = 512;
    int num_blocks_init = (num_runs + threads_per_block_init - 1) / threads_per_block_init;

    printf("Initializing device RNG states...\n");
    init_rng_states_kernel<<<num_blocks_init, threads_per_block_init>>>(rng_states, seed, num_runs);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for errors from the init kernel launch
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // Wait for init kernel

    printf("RNG states initialized.\n");

    // --- Simulation Kernel Launch ---
    int threads_per_block = 256;
    int num_blocks = (num_runs + threads_per_block - 1) / threads_per_block;

    printf("Launching simulation kernel with %d blocks and %d threads per block for %d runs.\n", num_blocks, threads_per_block, num_runs);

    simulate_levy_runs_kernel<<<num_blocks, threads_per_block>>>(
        num_runs,
        max_time,
        alpha, beta, mu, c, delta_t,
        vision_radius_sq,
        boundary,
        d_targets,
        num_targets,
        middle_index, // Pass starting target index
        d_starting_pos,
        d_first_return_time,
        d_first_hit_time,
        rng_states // Pass the device array of initialized states
    );

    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    // Wait for the kernel to complete
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    printf("Simulation kernel finished.\n");

    // --- Retrieve Results ---
    int* h_first_return_time = (int*)malloc(num_runs * sizeof(int));
    int* h_first_hit_time = (int*)malloc(num_runs * sizeof(int));

    if (h_first_return_time == NULL || h_first_hit_time == NULL) {
        fprintf(stderr, "Failed to allocate host memory for results.\n");
        // Clean up device memory before exiting
        cudaFree(d_targets);
        cudaFree(d_starting_pos);
        cudaFree(d_first_return_time);
        cudaFree(d_first_hit_time);
        cudaFree(rng_states);
        free(targets);
        free(h_first_return_time); // Free if one failed but the other succeeded
        free(h_first_hit_time);
        return 1;
    }


    CHECK_CUDA_ERROR(cudaMemcpy(h_first_return_time, d_first_return_time, num_runs * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_first_hit_time, d_first_hit_time, num_runs * sizeof(int), cudaMemcpyDeviceToHost));

    // --- Process Results (optional, matching Python's print) ---
    printf("\nSimulation Results Summary (first 10 and last 10 runs):\n");
    for (int i = 0; i < num_runs; ++i) {
        // Only print a few to avoid excessive output, or print a summary
        if (i < 10 || i >= num_runs - 10) {
             printf("Run %d: First Return = %d, First Hit = %d\n",
                i, h_first_return_time[i], h_first_hit_time[i]);
        }
    }

    // You could save h_first_return_time and h_first_hit_time to a file here,
    // similar to the Python pickle.
    // --- Save Results to Binary File ---
    const char* filename = "simulation_results.pickle";
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open file %s for writing.\n", filename);
    } else {
        // Write the total number of runs first
        fwrite(&num_runs, sizeof(int), 1, fp);

        // Then write each run's data
        for (int i = 0; i < num_runs; ++i) {
            RunData rd;
            rd.run_index = i;
            rd.first_return_time = h_first_return_time[i];
            rd.first_hit_time = h_first_hit_time[i];

            fwrite(&rd, sizeof(RunData), 1, fp);
        }

        fclose(fp);
        printf("Results saved to %s\n", filename);
    }
    // --- Cleanup ---
    cudaFree(d_targets);
    cudaFree(d_starting_pos);
    cudaFree(d_first_return_time);
    cudaFree(d_first_hit_time);
    cudaFree(rng_states); // Free cuRAND states device memory

    // No host generator handle is created/destroyed in this version with init_rng_states_kernel

    free(targets);
    free(h_first_return_time);
    free(h_first_hit_time);

    printf("\nCleanup complete. Simulation finished.\n");

    return 0;
}
