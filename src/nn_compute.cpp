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

    // Bind shader program
    glUseProgram(m_computeProgram);

    // Bind buffers
    m_buffers->bindBuffers(0, 1, 2);

    // Start profiling if enabled
    if (m_profilingEnabled && m_timerQuery) {
        glBeginQuery(GL_TIME_ELAPSED, m_timerQuery);
    }

    const auto& layerInfo = m_buffers->getLayerInfo();

    // Dispatch compute shader for each layer
    for (size_t i = 0; i < layerInfo.size(); ++i) {
        // Set layer index uniform
        GLint layerLoc = glGetUniformLocation(m_computeProgram, "u_layerIndex");
        glUniform1ui(layerLoc, static_cast<GLuint>(i));

        // Calculate work groups (1D dispatch)
        uint32_t outputSize = layerInfo[i].outputSize;
        uint32_t workGroupSize = 256;  // Must match shader local_size_x
        uint32_t workGroups = (outputSize + workGroupSize - 1) / workGroupSize;

        // Dispatch
        glDispatchCompute(workGroups, 1, 1);

        // Memory barrier - critical!
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // End profiling
    if (m_profilingEnabled && m_timerQuery) {
        glEndQuery(GL_TIME_ELAPSED);

        // Retrieve timing (may stall - only for debugging!)
        GLuint64 timeElapsed;
        glGetQueryObjectui64v(m_timerQuery, GL_QUERY_RESULT, &timeElapsed);
        m_lastExecutionTimeMs = static_cast<float>(timeElapsed) / 1000000.0f;  // ns to ms
    }
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