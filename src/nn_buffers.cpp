#include "nn_buffers.h"
#include <iostream>
#include <numeric>

NeuralBuffers::~NeuralBuffers() {
    cleanup();
}

void NeuralBuffers::initialize(const std::vector<uint32_t>& layerSizes,
                                const std::vector<uint32_t>& activations) {
    cleanup();  // Clean up any existing buffers

    // Verify struct size matches std140 expectations (must be 32 bytes)
    static_assert(sizeof(LayerInfo) == 32, "LayerInfo must be 32 bytes for std140 alignment");

    m_topology = layerSizes;

    // Compute total neurons
    m_totalNeurons = std::accumulate(layerSizes.begin(), layerSizes.end(), 0u);

    // Compute offsets and layer info
    computeOffsets();

    // Store activation types
    if (activations.size() != layerSizes.size() - 1) {
        std::cerr << "[ERROR] Activation count mismatch. Expected "
                  << (layerSizes.size() - 1) << ", got " << activations.size() << "\n";
        return;
    }

    for (size_t i = 0; i < activations.size(); ++i) {
        m_layerInfo[i].activationType = activations[i];
    }

    // Create GPU buffers
    createBuffers();

    std::cout << "[INFO] Neural buffers initialized:\n";
    std::cout << "  Total neurons: " << m_totalNeurons << "\n";
    std::cout << "  Total weights: " << m_totalWeights << "\n";
    std::cout << "  Total biases:  " << m_totalBiases << "\n";
}

void NeuralBuffers::computeOffsets() {
    m_layerInfo.clear();
    m_totalWeights = 0;
    m_totalBiases = 0;

    uint32_t activationOffset = 0;  // Track position in activations buffer

    for (size_t i = 0; i < m_topology.size() - 1; ++i) {
        LayerInfo info;
        info.inputSize = m_topology[i];
        info.outputSize = m_topology[i + 1];
        info.weightOffset = m_totalWeights;
        info.biasOffset = m_totalBiases;
        info.activationType = 0;  // Will be set later

        // Calculate activation buffer offsets
        info.inputOffset = activationOffset;
        info.outputOffset = activationOffset + info.inputSize;

        // Initialize padding
        info._padding[0] = 0;

        m_layerInfo.push_back(info);

        // Debug: Print layer info
        std::cout << "  Layer " << i << ": "
                  << "in=" << info.inputSize << " out=" << info.outputSize
                  << " | inputOff=" << info.inputOffset
                  << " outputOff=" << info.outputOffset << "\n";

        // Move offset forward by the number of neurons in this layer's input
        activationOffset += m_topology[i];

        // Weights: inputSize * outputSize
        m_totalWeights += info.inputSize * info.outputSize;

        // Biases: outputSize
        m_totalBiases += info.outputSize;
    }
}

void NeuralBuffers::createBuffers() {
    // Weights SSBO
    glGenBuffers(1, &m_weightsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_weightsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 m_totalWeights * sizeof(float),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    // Biases SSBO
    glGenBuffers(1, &m_biasesSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_biasesSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 m_totalBiases * sizeof(float),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    // Activations SSBO
    glGenBuffers(1, &m_activationsSSBO);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_activationsSSBO);
    glBufferData(GL_SHADER_STORAGE_BUFFER,
                 m_totalNeurons * sizeof(float),
                 nullptr,
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::uploadWeights(const std::vector<float>& weights) {
    if (weights.size() != m_totalWeights) {
        std::cerr << "[ERROR] Weight count mismatch. Expected "
                  << m_totalWeights << ", got " << weights.size() << "\n";
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_weightsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    weights.size() * sizeof(float),
                    weights.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::uploadBiases(const std::vector<float>& biases) {
    if (biases.size() != m_totalBiases) {
        std::cerr << "[ERROR] Bias count mismatch. Expected "
                  << m_totalBiases << ", got " << biases.size() << "\n";
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_biasesSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    biases.size() * sizeof(float),
                    biases.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::setInputs(const std::vector<float>& inputs) {
    if (inputs.size() != m_topology[0]) {
        std::cerr << "[ERROR] Input size mismatch. Expected "
                  << m_topology[0] << ", got " << inputs.size() << "\n";
        return;
    }

    // Write to beginning of activations buffer (input layer)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_activationsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    inputs.size() * sizeof(float),
                    inputs.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::clearActivations() {
    // Zero out entire activation buffer
    std::vector<float> zeros(m_totalNeurons, 0.0f);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_activationsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    m_totalNeurons * sizeof(float),
                    zeros.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::uploadActivations(const std::vector<float>& activations) {
    if (activations.size() != m_totalNeurons) {
        std::cerr << "[ERROR] Activation count mismatch. Expected "
                  << m_totalNeurons << ", got " << activations.size() << "\n";
        return;
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_activationsSSBO);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                    activations.size() * sizeof(float),
                    activations.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::readOutputs(std::vector<float>& outputs) const {
    uint32_t outputSize = m_topology.back();
    outputs.resize(outputSize);

    // Calculate offset to last layer
    uint32_t offset = m_totalNeurons - outputSize;

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_activationsSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER,
                       offset * sizeof(float),
                       outputSize * sizeof(float),
                       outputs.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::readAllActivations(std::vector<float>& activations) const {
    activations.resize(m_totalNeurons);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_activationsSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       m_totalNeurons * sizeof(float),
                       activations.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::readWeights(std::vector<float>& weights) const {
    weights.resize(m_totalWeights);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, m_weightsSSBO);
    glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0,
                       m_totalWeights * sizeof(float),
                       weights.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void NeuralBuffers::bindBuffers(GLuint weightsBinding,
                                 GLuint biasesBinding,
                                 GLuint activationsBinding) const {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, weightsBinding, m_weightsSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, biasesBinding, m_biasesSSBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, activationsBinding, m_activationsSSBO);
}

void NeuralBuffers::cleanup() {
    if (m_weightsSSBO) {
        glDeleteBuffers(1, &m_weightsSSBO);
        m_weightsSSBO = 0;
    }
    if (m_biasesSSBO) {
        glDeleteBuffers(1, &m_biasesSSBO);
        m_biasesSSBO = 0;
    }
    if (m_activationsSSBO) {
        glDeleteBuffers(1, &m_activationsSSBO);
        m_activationsSSBO = 0;
    }
}