# GPU-Based Neural Network Visualizer (OpenGL)

## Overview
This project is a real-time, GPU-accelerated neural network visualizer built using modern OpenGL (4.6) and C++. Its primary goal is to execute neural network inference directly on the GPU using compute shaders and visualize internal network states (activations, weights, gradients) in an interactive and intuitive way.

Unlike typical ML tooling, this project:
- Does not rely on CUDA, OpenCL, or external ML frameworks
- Uses pure OpenGL compute shaders
- Treats the neural network as a first-class graphics object

The result is a hybrid AI + graphics debugging tool and a deep exploration of GPU programming.

## Key Objectives
- Implement a GPU-resident neural network
- Perform forward inference entirely on the GPU
- Visualize:
    - Neuron activations
    - Weight magnitudes and signs
    - Network topology
- Provide interactive inspection of network internals
- Serve as a learning, research, and portfolio-grade project

## Core Features

### Neural Network Engine
- **Feedforward Multi-Layer Perceptron (MLP)**
    - Arbitrary layer sizes
    - Static or dynamic topology
    - Activation functions: ReLU, Sigmoid, Tanh (Extensible)

### GPU Compute
- Compute shaders for:
    - Matrix multiplication
    - Bias addition
    - Activation evaluation
- SSBO-based data storage
- One dispatch per layer
- Zero CPU-side math during inference

### Visualization
- Neurons rendered as instanced primitives
- Color encoding:
    - Activation value
    - Gradient magnitude (future)
- Connections rendered as weighted edges
- Real-time updates per inference step

### Interaction (Planned / In Progress)
- Pause / step inference
- Inspect individual neurons
- Toggle visualization modes
- Live parameter tweaking

## High-Level Architecture

### CPU (C++)
1. Window & Input (GLFW)
2. UI Layer (ImGui – optional)
3. Network Configuration
4. Compute Dispatch Control

### GPU (OpenGL 4.6)
5. SSBO: Weights
6. SSBO: Biases
7. SSBO: Activations
8. Compute Shader: Forward Pass
9. Render Pipeline: Visualization

## Data-Oriented Design

### GPU Buffers

#### Weights SSBO
- Flat array with per-layer offsets
- **std430 layout** for explicit alignment
- Pre-computed offsets passed as uniforms

#### Biases SSBO
- Linear memory layout per layer

#### Activations SSBO
- Updated every inference step
- **Consider double-buffering** (ping-pong) for async compute while rendering

### Rationale
- Linear memory layout for cache efficiency
- Explicit offsets for deterministic access
- Designed to scale to large networks
- Alignment rules critical for GPU correctness

## Technology Stack

### Language
- **C++20**

### Graphics & Compute
- **OpenGL 4.6** (Core Profile)
- **GLSL 4.60**
- **Compute Shaders**
- **Shader Storage Buffer Objects (SSBOs)**

### Windowing & Context
- **GLFW**

### OpenGL Loader
- **GLAD** (custom-generated for OpenGL 4.6)

### Math
- **GLM**

### UI (Optional / Later Phase)
- **Dear ImGui**

### Build System
- **CMake** (≥ 3.26)

### Development Environment
- **CLion**
- **Windows 11**
- **Native GPU drivers** (NVIDIA / AMD / Intel)

### Debugging & Profiling
- **OpenGL Debug Output** (enable from day 1!)
- **RenderDoc**
- **GPU vendor tools** (Nsight / Radeon GPU Profiler)
- **GPU timer queries** for performance profiling

## Project Structure

```
nn-visualizer/
├── CMakeLists.txt
├── README.md
├── config/
│   └── network_config.json      # Network topology configurations
├── external/
│   ├── glad/
│   ├── glfw/
│   └── glm/
├── src/
│   ├── main.cpp
│   ├── gl_context.cpp
│   ├── nn_compute.cpp
│   ├── nn_buffers.cpp
│   ├── renderer.cpp
│   ├── camera.cpp               # Orbital camera system
│   └── shader_loader.cpp        # Hot shader reload
├── shaders/
│   ├── forward.comp
│   ├── neuron.vert
│   ├── neuron.frag
│   └── colormap.glsl            # Perceptually uniform colormaps
├── tests/
│   ├── buffer_tests.cpp         # SSBO layout validation
│   └── cpu_reference.cpp        # CPU reference implementation
└── assets/
    └── golden_images/           # Visual regression test data
```

## Development Roadmap

### Phase 0 — Spike Solution (Recommended Start)
**Goal:** Validate entire pipeline in 1-2 days
- 2-layer XOR network (2→2→1)
- Hardcoded weights/biases
- Single compute shader
- Point sprite visualization
- **Critical:** Proves the architecture works before committing

### Phase 1 — Core (MVP)
- OpenGL context creation with debug output enabled
- SSBO layout definition with std430 alignment
- GPU forward pass for configurable network
- Minimal visualization (instanced point sprites)
- Configuration file loading (JSON)
- CPU reference validation for correctness

### Phase 2 — Interactivity
- Layer navigation
- Activation inspection
- Runtime parameter changes
- Orbital camera system
- Hot shader reload
- Performance profiling UI

### Phase 3 — Training (Advanced)
- Backpropagation via compute shaders
- Gradient visualization
- Loss tracking
- Memory barriers and synchronization

### Phase 4 — Extensions
- CNN layers
- Feature map visualization
- Temporal activation playback
- Level-of-detail for large networks

## Implementation Priorities

### Critical Path for Phase 1:
1. ✅ Hello Triangle (verify OpenGL setup)
2. ✅ Single compute shader that squares numbers (verify compute pipeline)
3. ✅ 2-layer network with hardcoded topology
4. ✅ Point sprite visualization
5. ✅ Configuration system
6. ✅ THEN expand to arbitrary layers

### Don't Do This in Phase 1:
- ❌ Dynamic topology changes
- ❌ Connection visualization
- ❌ Complex activation functions
- ❌ Training/backprop

## Architecture & Design Guidelines

### SSBO Memory Layout (Critical)

```glsl
// Weights SSBO - std430 layout
layout(std430, binding = 0) buffer Weights {
    float weights[];
};

// Always use memory barriers after compute
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
```

**Key considerations:**
- Explicit std430 alignment rules
- Pre-compute offsets on CPU
- Atomic operations for future concurrent updates
- Double-buffering for async compute/render

### Compute Shader Optimization

```glsl
// Start with this work group size, then profile
layout(local_size_x = 256) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    // Add early-out for dead neurons
    if (activations[idx] == 0.0 && isReLU) return;
    // ...
}
```

**Optimization strategies:**
- Profile different work group sizes (64, 128, 256, 512)
- Consider tiling for matrix multiply (shared memory)
- Add early-out branches for sparse activations

### Network Topology Representation

```cpp
// Store layer metadata in uniform buffer, not SSBO
struct LayerInfo {
    uint inputSize;
    uint outputSize;
    uint weightOffset;
    uint biasOffset;
    uint activationType; // 0=ReLU, 1=Sigmoid, 2=Tanh
};
```

**Design principles:**
- Uniform buffers for small, frequently-accessed data
- SSBOs for large, bulk data
- Pre-compute all offsets on CPU

## Visualization System

### Color Mapping
- Use perceptually uniform colormaps (viridis, plasma)
- Consider colorblind-friendly palettes
- Don't use raw RGB for activation values

### Neuron Rendering
- Instanced rendering with transform feedback
- Billboard sprites initially
- Size proportional to activation magnitude
- Geometry shader upgrade for advanced shapes

### Camera System
- Orbital camera (arcball) essential for 3D networks
- Orthographic projection for 2D view
- Save/load camera positions
- Smooth interpolation between views

## Configuration System

### network_config.json
```json
{
  "name": "XOR Network",
  "layers": [2, 2, 1],
  "activations": ["relu", "sigmoid"],
  "weights_init": "xavier",
  "visualization": {
    "neuron_size": 0.1,
    "colormap": "viridis",
    "show_connections": false
  }
}
```

## Testing Strategy

### Unit Tests
- Buffer offset calculations
- Layer size mathematics
- SSBO layout correctness (CPU write, GPU read verification)

### Visual Regression Tests
- Capture framebuffer after fixed inference
- Compare against golden images
- Automatically detect rendering bugs

### Performance Benchmarks
Document GPU vs CPU speedup:
- Small network (784→128→10): **Target: >100x**
- Medium (784→512→256→10): **Target: >500x**
- Large (784→1024→512→256→10): **Target: >1000x**

## Critical Implementation Notes

### 1. OpenGL Debug Output (Enable First!)
```cpp
glEnable(GL_DEBUG_OUTPUT);
glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
glDebugMessageCallback(debugCallback, nullptr);
```
**This saves hours of debugging cryptic errors.**

### 2. Memory Barriers (Don't Forget!)
```cpp
glDispatchCompute(workGroupsX, 1, 1);
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
// Now safe to read results
```
**Forgetting this = reading stale data.**

### 3. GPU Timer Queries
```cpp
glBeginQuery(GL_TIME_ELAPSED, query);
glDispatchCompute(...);
glEndQuery(GL_TIME_ELAPSED);
// Retrieve timing data for profiling
```

### 4. Shader Compilation Errors
- Log full info logs
- Add `#line` directives for better error messages
- GLSL errors are cryptic by default

## Performance Considerations

### Biggest Risks & Mitigations

1. **Matrix Multiply Performance**
    - **Risk:** Naive GLSL implementation too slow
    - **Mitigation:** Prototype separately, use tiling, consider hand-optimization

2. **Large Network Visualization**
    - **Risk:** 10k+ neurons won't render at interactive framerates
    - **Mitigation:** Level-of-detail system, culling, instancing

3. **Memory Limits**
    - **Risk:** SSBO size limits vary by GPU
    - **Mitigation:** Test on multiple GPUs, add memory usage UI

### Optimization Checklist
- [ ] Profile work group sizes
- [ ] Implement shared memory tiling for matmul
- [ ] Add frustum culling for neurons
- [ ] Use instanced rendering everywhere
- [ ] Minimize CPU-GPU synchronization
- [ ] Double-buffer where possible

## Why This Project Matters

- Demonstrates **deep GPU programming knowledge**
- Bridges **graphics and machine learning**
- Uses OpenGL in a **non-traditional, compute-heavy way**
- Shows strong understanding of:
    - Data-oriented design
    - Shader programming
    - Performance constraints
    - Real-time rendering
- **Rare and technically impressive** portfolio project
- Practical tool for understanding neural network internals

## Requirements

- **Windows 11**
- **GPU with OpenGL 4.6 support**
- **Updated graphics drivers**
- **CMake ≥ 3.26**
- **C++20-compatible compiler**

## Build Instructions

```bash
# Clone repository
git clone https://github.com/YassineKaibi/NeuraVis.git
cd nn-visualizer

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
cmake --build . --config Release

# Run
./nn-visualizer
```

## CMake Improvements

```cmake
# Shader compilation as build step
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/shaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory 
          ${CMAKE_SOURCE_DIR}/shaders 
          ${CMAKE_BINARY_DIR}/shaders
  DEPENDS ${CMAKE_SOURCE_DIR}/shaders
)

# Enable warnings
target_compile_options(nn-visualizer PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# Link-time optimization for Release
set_target_properties(nn-visualizer PROPERTIES
  INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
)
```

## Usage

```bash
# Run with default XOR network
./nn-visualizer

# Load custom configuration
./nn-visualizer --config config/mnist_network.json

# Enable profiling
./nn-visualizer --profile

# Start with CPU validation
./nn-visualizer --validate
```

## Validation Mode

CPU reference implementation ensures correctness:
```bash
./nn-visualizer --validate --config config/small_network.json
```
Computes inference on both CPU and GPU, reports differences.

## Documentation

### Blog Post Topics
- "Implementing Neural Networks in Pure GLSL"
- "GPU Memory Layout for ML: A Deep Dive"
- "Visualizing the Invisible: Making Neural Networks Tangible"
- "Compute Shaders vs CUDA: A Case Study"

## Contributing

This is primarily a learning and portfolio project, but suggestions and improvements are welcome!

Areas for contribution:
- Additional activation functions
- Improved visualization techniques
- Performance optimizations
- Cross-platform support (Linux, macOS)
- Network architecture presets

## License

[Your chosen license]

## Acknowledgments

- OpenGL compute shader tutorials
- Neural network visualization research papers
- Open-source graphics community

## Contact

[https://www.linkedin.com/in/mohamedyassinekaibi/]

---

**Note:** This is an evolving project.