#include "nn_compute.h"
#include "shader_loader.h"
#include <iostream>

NeuralCompute::~NeuralCompute() {
    cleanup();
}

bool NeuralCompute::initialize(const std::string& computeShaderPath, NeuralBuffers& buffers) {
    m_buffers = &buffers;

    // Load compute shader
    m_computeProgram = ShaderLoader::loadComputeShader(computeShaderPath);
    if (m_computeProgram == 0) {
        std::cerr << "[ERROR] Failed to load compute shader: " << computeShaderPath << "\n";
        return false;
    }

    // Create uniform buffer for layer info
    glGenBuffers(1, &m_layerInfoUBO);
    uploadLayerInfo();

    // Create timer query for profiling
    if (m_profilingEnabled) {
        glGenQueries(1, &m_timerQuery);
    }

    std::cout << "[INFO] Neural compute initialized\n";
    return true;
}

void NeuralCompute::forward() {
    if (!m_buffers || m_computeProgram == 0) {
        std::cerr << "[ERROR] NeuralCompute not initialized\n";
        return;
    }

    const auto& layerInfo = m_buffers->getLayerInfo();

    // Dispatch compute shader for each layer
    for (size_t i = 0; i < layerInfo.size(); ++i) {
        forwardLayer(i);
    }
}

void NeuralCompute::forwardLayer(size_t layerIndex) {
    if (!m_buffers || m_computeProgram == 0) {
        std::cerr << "[ERROR] NeuralCompute not initialized\n";
        return;
    }

    const auto& layerInfo = m_buffers->getLayerInfo();

    if (layerIndex >= layerInfo.size()) {
        std::cerr << "[ERROR] Layer index out of bounds: " << layerIndex << "\n";
        return;
    }

    // Bind shader program
    glUseProgram(m_computeProgram);

    // Bind buffers
    m_buffers->bindBuffers(0, 1, 2);

    // Bind UBO with layer info
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_layerInfoUBO);

    // Set layer index uniform
    GLint layerLoc = glGetUniformLocation(m_computeProgram, "u_layerIndex");
    glUniform1ui(layerLoc, static_cast<GLuint>(layerIndex));

    // Calculate work groups (1D dispatch)
    uint32_t outputSize = layerInfo[layerIndex].outputSize;
    uint32_t workGroupSize = 256;  // Must match shader local_size_x
    uint32_t workGroups = (outputSize + workGroupSize - 1) / workGroupSize;

    // Dispatch
    glDispatchCompute(workGroups, 1, 1);

    // Memory barrier - critical!
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

size_t NeuralCompute::getLayerCount() const {
    if (!m_buffers) return 0;
    return m_buffers->getLayerInfo().size();
}

void NeuralCompute::uploadLayerInfo() {
    const auto& layerInfo = m_buffers->getLayerInfo();

    glBindBuffer(GL_UNIFORM_BUFFER, m_layerInfoUBO);
    glBufferData(GL_UNIFORM_BUFFER,
                 layerInfo.size() * sizeof(NeuralBuffers::LayerInfo),
                 layerInfo.data(),
                 GL_STATIC_DRAW);

    // Bind to binding point 0
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_layerInfoUBO);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void NeuralCompute::cleanup() {
    if (m_computeProgram) {
        glDeleteProgram(m_computeProgram);
        m_computeProgram = 0;
    }
    if (m_layerInfoUBO) {
        glDeleteBuffers(1, &m_layerInfoUBO);
        m_layerInfoUBO = 0;
    }
    if (m_timerQuery) {
        glDeleteQueries(1, &m_timerQuery);
        m_timerQuery = 0;
    }
}