#pragma once

#include <glad/glad.h>
#include <vector>
#include <cstdint>

/**
 * @brief Manages GPU buffers (SSBOs) for neural network data
 *
 * Layout Strategy:
 * - Weights: Flat array with per-layer offsets
 * - Biases: Flat array with per-layer offsets
 * - Activations: Double-buffered for async compute/render
 *
 * Memory Layout (std430):
 * - Explicit alignment rules enforced
 * - Pre-computed offsets for deterministic access
 */
class NeuralBuffers {
public:
    struct LayerInfo {
        uint32_t inputSize;
        uint32_t outputSize;
        uint32_t weightOffset;     // Offset into weights buffer (in floats)
        uint32_t biasOffset;       // Offset into biases buffer (in floats)
        uint32_t activationType;   // 0=ReLU, 1=Sigmoid, 2=Tanh
        uint32_t inputOffset;      // Offset into activations buffer for inputs
        uint32_t outputOffset;     // Offset into activations buffer for outputs
        uint32_t _padding[1];      // Align to 32 bytes for std140
    };

    NeuralBuffers() = default;
    ~NeuralBuffers();

    // Prevent copying
    NeuralBuffers(const NeuralBuffers&) = delete;
    NeuralBuffers& operator=(const NeuralBuffers&) = delete;

    /**
     * @brief Initialize buffers for a network topology
     * @param layerSizes Size of each layer (e.g., {2, 2, 1} for XOR)
     * @param activations Activation type per layer (same length as layerSizes - 1)
     */
    void initialize(const std::vector<uint32_t>& layerSizes,
                    const std::vector<uint32_t>& activations);

    /**
     * @brief Upload weight data to GPU
     * @param weights Flat array of all weights (concatenated per layer)
     */
    void uploadWeights(const std::vector<float>& weights);

    /**
     * @brief Upload bias data to GPU
     * @param biases Flat array of all biases (concatenated per layer)
     */
    void uploadBiases(const std::vector<float>& biases);

    /**
     * @brief Set input activations (first layer)
     * @param inputs Input values to network
     */
    void setInputs(const std::vector<float>& inputs);

    /**
     * @brief Clear all activations (zero out activation buffer)
     * Call this when switching inputs to reset network state
     */
    void clearActivations();

    /**
     * @brief Upload arbitrary activation values to GPU
     * Used for smooth animation interpolation
     */
    void uploadActivations(const std::vector<float>& activations);

    /**
     * @brief Read output activations (last layer) from GPU
     * @param outputs Vector to store output values
     */
    void readOutputs(std::vector<float>& outputs) const;

    /**
     * @brief Read all activations from GPU (for visualization)
     * @param activations Vector to store all activation values
     */
    void readAllActivations(std::vector<float>& activations) const;

    /**
     * @brief Read all weights from GPU (for connection visualization)
     * @param weights Vector to store all weight values
     */
    void readWeights(std::vector<float>& weights) const;

    /**
     * @brief Bind buffers to shader binding points
     * @param weightsBinding Binding point for weights SSBO
     * @param biasesBinding Binding point for biases SSBO
     * @param activationsBinding Binding point for activations SSBO
     */
    void bindBuffers(GLuint weightsBinding = 0,
                     GLuint biasesBinding = 1,
                     GLuint activationsBinding = 2) const;

    /**
     * @brief Get layer metadata for uploading to uniform buffer
     */
    const std::vector<LayerInfo>& getLayerInfo() const { return m_layerInfo; }

    /**
     * @brief Get total number of neurons across all layers
     */
    uint32_t getTotalNeuronCount() const { return m_totalNeurons; }

    /**
     * @brief Get network topology (layer sizes)
     */
    const std::vector<uint32_t>& getTopology() const { return m_topology; }

private:
    GLuint m_weightsSSBO = 0;
    GLuint m_biasesSSBO = 0;
    GLuint m_activationsSSBO = 0;

    std::vector<uint32_t> m_topology;       // Layer sizes (e.g., {2, 2, 1})
    std::vector<LayerInfo> m_layerInfo;     // Per-layer metadata

    uint32_t m_totalWeights = 0;
    uint32_t m_totalBiases = 0;
    uint32_t m_totalNeurons = 0;

    void computeOffsets();
    void createBuffers();
    void cleanup();
};