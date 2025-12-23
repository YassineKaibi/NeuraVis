#pragma once

#include "nn_buffers.h"
#include <glad/glad.h>
#include <vector>
#include <string>

/**
 * @brief Manages GPU compute shader execution for neural network inference
 *
 * Responsibilities:
 * - Load and compile compute shaders
 * - Dispatch compute work per layer
 * - Upload layer metadata to uniforms
 * - Synchronize GPU memory barriers
 */
class NeuralCompute {
public:
    NeuralCompute() = default;
    ~NeuralCompute();

    // Prevent copying
    NeuralCompute(const NeuralCompute&) = delete;
    NeuralCompute& operator=(const NeuralCompute&) = delete;

    /**
     * @brief Initialize compute pipeline
     * @param computeShaderPath Path to forward pass compute shader
     * @param buffers Reference to neural network buffers
     * @return true if initialization successful
     */
    bool initialize(const std::string& computeShaderPath, NeuralBuffers& buffers);

    /**
     * @brief Execute forward pass on GPU
     *
     * Dispatches compute shader for each layer sequentially
     * Inserts memory barriers between layers
     */
    void forward();

    /**
     * @brief Execute forward pass for a single layer
     * @param layerIndex Index of the layer to compute (0-based)
     */
    void forwardLayer(size_t layerIndex);

    /**
     * @brief Get the total number of layers in the network
     */
    size_t getLayerCount() const;

    /**
     * @brief Get last recorded GPU execution time in milliseconds
     */
    float getLastExecutionTime() const { return m_lastExecutionTimeMs; }

    /**
     * @brief Enable/disable GPU profiling with timer queries
     */
    void setProfilingEnabled(bool enabled) { m_profilingEnabled = enabled; }

private:
    GLuint m_computeProgram = 0;
    GLuint m_layerInfoUBO = 0;          // Uniform buffer for layer metadata
    GLuint m_timerQuery = 0;            // GPU timer query for profiling

    NeuralBuffers* m_buffers = nullptr;
    bool m_profilingEnabled = false;
    float m_lastExecutionTimeMs = 0.0f;

    void uploadLayerInfo();
    void cleanup();
};