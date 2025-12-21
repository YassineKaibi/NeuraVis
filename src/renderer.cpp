#include "renderer.h"
#include "shader_loader.h"
#include <iostream>
#include <cmath>

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::initialize(NeuralBuffers& buffers) {
    m_buffers = &buffers;
    m_totalNeurons = buffers.getTotalNeuronCount();

    // Load shaders
    m_neuronProgram = ShaderLoader::loadShaderProgram("shaders/neuron.vert",
                                                       "shaders/neuron.frag");
    if (m_neuronProgram == 0) {
        std::cerr << "[ERROR] Failed to load neuron shaders\n";
        return false;
    }

    // Generate 3D layout for neurons
    generateNeuronLayout();

    // Create VAO and VBO
    glGenVertexArrays(1, &m_neuronVAO);
    glGenBuffers(1, &m_neuronPositionVBO);

    uploadNeuronPositions();

    std::cout << "[INFO] Renderer initialized with " << m_totalNeurons << " neurons\n";
    return true;
}

void Renderer::render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix) {
    if (!m_buffers || m_neuronProgram == 0) {
        return;
    }

    glUseProgram(m_neuronProgram);

    // Upload matrices
    GLint viewLoc = glGetUniformLocation(m_neuronProgram, "u_view");
    GLint projLoc = glGetUniformLocation(m_neuronProgram, "u_projection");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &viewMatrix[0][0]);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projMatrix[0][0]);

    // Upload visualization config
    GLint neuronSizeLoc = glGetUniformLocation(m_neuronProgram, "u_neuronSize");
    glUniform1f(neuronSizeLoc, m_config.neuronSize);

    // Bind activations SSBO for reading in shader
    m_buffers->bindBuffers(0, 1, 2);

    // Draw instanced points
    glBindVertexArray(m_neuronVAO);
    glDrawArrays(GL_POINTS, 0, m_totalNeurons);
    glBindVertexArray(0);
}

void Renderer::generateNeuronLayout() {
    m_neuronPositions.clear();
    m_neuronPositions.reserve(m_totalNeurons);

    const auto& topology = m_buffers->getTopology();
    float layerSpacing = 3.0f;

    for (size_t layerIdx = 0; layerIdx < topology.size(); ++layerIdx) {
        uint32_t layerSize = topology[layerIdx];
        float neuronSpacing = 1.0f;

        // Center the layer vertically
        float yOffset = -(layerSize - 1) * neuronSpacing / 2.0f;

        for (uint32_t neuronIdx = 0; neuronIdx < layerSize; ++neuronIdx) {
            float x = layerIdx * layerSpacing;
            float y = yOffset + neuronIdx * neuronSpacing;
            float z = 0.0f;

            m_neuronPositions.emplace_back(x, y, z);
        }
    }
}

void Renderer::uploadNeuronPositions() {
    glBindVertexArray(m_neuronVAO);

    glBindBuffer(GL_ARRAY_BUFFER, m_neuronPositionVBO);
    glBufferData(GL_ARRAY_BUFFER,
                 m_neuronPositions.size() * sizeof(glm::vec3),
                 m_neuronPositions.data(),
                 GL_STATIC_DRAW);

    // Position attribute (location = 0)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);

    glBindVertexArray(0);
}

void Renderer::cleanup() {
    if (m_neuronVAO) {
        glDeleteVertexArrays(1, &m_neuronVAO);
        m_neuronVAO = 0;
    }
    if (m_neuronPositionVBO) {
        glDeleteBuffers(1, &m_neuronPositionVBO);
        m_neuronPositionVBO = 0;
    }
    if (m_neuronProgram) {
        glDeleteProgram(m_neuronProgram);
        m_neuronProgram = 0;
    }
}