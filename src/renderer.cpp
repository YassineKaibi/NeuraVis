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

    // Load neuron shaders
    m_neuronProgram = ShaderLoader::loadShaderProgram("shaders/neuron.vert",
                                                       "shaders/neuron.frag");
    if (m_neuronProgram == 0) {
        std::cerr << "[ERROR] Failed to load neuron shaders\n";
        return false;
    }

    // Load connection shaders (with geometry shader for thick lines)
    m_connectionProgram = ShaderLoader::loadShaderProgram("shaders/connection.vert",
                                                           "shaders/connection.geom",
                                                           "shaders/connection.frag");
    if (m_connectionProgram == 0) {
        std::cerr << "[ERROR] Failed to load connection shaders\n";
        return false;
    }

    // Generate 3D layout for neurons
    generateNeuronLayout();

    // Create VAO and VBO for neurons
    glGenVertexArrays(1, &m_neuronVAO);
    glGenBuffers(1, &m_neuronPositionVBO);
    uploadNeuronPositions();

    // Generate and upload connections
    generateConnections();
    glGenVertexArrays(1, &m_connectionVAO);
    glGenBuffers(1, &m_connectionVBO);
    uploadConnections();

    std::cout << "[INFO] Renderer initialized with " << m_totalNeurons << " neurons and "
              << m_connectionCount << " connections\n";
    return true;
}

void Renderer::render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix) {
    if (!m_buffers || m_neuronProgram == 0) {
        std::cerr << "[ERROR] Renderer not properly initialized\n";
        return;
    }

    static bool printedOnce = false;
    if (!printedOnce) {
        std::cout << "[DEBUG] First render - View matrix:\n";
        for (int i = 0; i < 4; i++) {
            std::cout << "  [" << viewMatrix[i][0] << ", " << viewMatrix[i][1] << ", "
                      << viewMatrix[i][2] << ", " << viewMatrix[i][3] << "]\n";
        }
        std::cout << "[DEBUG] Projection matrix:\n";
        for (int i = 0; i < 4; i++) {
            std::cout << "  [" << projMatrix[i][0] << ", " << projMatrix[i][1] << ", "
                      << projMatrix[i][2] << ", " << projMatrix[i][3] << "]\n";
        }
        printedOnce = true;
    }

    // Render connections first (behind neurons)
    if (m_config.showConnections && m_connectionProgram != 0) {
        renderConnections(viewMatrix, projMatrix);
    }

    // Render neurons on top
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

    // Draw points
    glBindVertexArray(m_neuronVAO);

    static bool printedDrawInfo = false;
    if (!printedDrawInfo) {
        std::cout << "[DEBUG] Drawing " << m_totalNeurons << " neurons as GL_POINTS\n";
        std::cout << "[DEBUG] Neuron size uniform: " << m_config.neuronSize << "\n";

        // Check VAO state
        GLint vaoBinding;
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vaoBinding);
        std::cout << "[DEBUG] Current VAO: " << vaoBinding << " (should be " << m_neuronVAO << ")\n";

        printedDrawInfo = true;
    }

    glDrawArrays(GL_POINTS, 0, m_totalNeurons);
    glBindVertexArray(0);

    // Check for OpenGL errors
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "[ERROR] OpenGL error after neuron render: " << err << "\n";
    }
}

void Renderer::generateNeuronLayout() {
    m_neuronPositions.clear();
    m_neuronPositions.reserve(m_totalNeurons);

    const auto& topology = m_buffers->getTopology();
    float layerSpacing = 3.0f;

    std::cout << "[DEBUG] Generating neuron positions:\n";

    for (size_t layerIdx = 0; layerIdx < topology.size(); ++layerIdx) {
        uint32_t layerSize = topology[layerIdx];
        float neuronSpacing = 1.0f;

        // Center the layer vertically (cast to float BEFORE negation to avoid unsigned wraparound)
        float yOffset = -(static_cast<float>(layerSize - 1) * neuronSpacing) / 2.0f;

        std::cout << "  Layer " << layerIdx << " (" << layerSize << " neurons):\n";

        for (uint32_t neuronIdx = 0; neuronIdx < layerSize; ++neuronIdx) {
            float x = layerIdx * layerSpacing;
            float y = yOffset + neuronIdx * neuronSpacing;
            float z = 0.0f;

            m_neuronPositions.emplace_back(x, y, z);
            std::cout << "    Neuron " << (m_neuronPositions.size() - 1)
                      << ": (" << x << ", " << y << ", " << z << ")\n";
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

    // Verify upload
    std::cout << "[DEBUG] Uploaded " << m_neuronPositions.size() << " neuron positions to VBO\n";
    std::cout << "[DEBUG] VBO size: " << (m_neuronPositions.size() * sizeof(glm::vec3)) << " bytes\n";
}

void Renderer::generateConnections() {
    m_connectionVertices.clear();

    const auto& topology = m_buffers->getTopology();
    const auto& layerInfo = m_buffers->getLayerInfo();

    // Read weights from GPU
    std::vector<float> weights;
    m_buffers->readWeights(weights);

    std::cout << "[DEBUG] Generating connections (" << weights.size() << " weights total):\n";

    uint32_t neuronOffset = 0;
    uint32_t weightOffset = 0;
    uint32_t connectionCount = 0;

    for (size_t layerIdx = 0; layerIdx < topology.size() - 1; ++layerIdx) {
        uint32_t inputSize = topology[layerIdx];
        uint32_t outputSize = topology[layerIdx + 1];

        // Iterate through each output neuron
        for (uint32_t outIdx = 0; outIdx < outputSize; ++outIdx) {
            // Connect to each input neuron
            for (uint32_t inIdx = 0; inIdx < inputSize; ++inIdx) {
                // Read actual weight value from buffer
                // Weight layout: weights[layer][out_neuron][in_neuron]
                uint32_t weightIdx = weightOffset + (outIdx * inputSize) + inIdx;
                float weight = weights[weightIdx];

                glm::vec3 startPos = m_neuronPositions[neuronOffset + inIdx];
                glm::vec3 endPos = m_neuronPositions[neuronOffset + inputSize + outIdx];

                // For GL_LINES, we need 2 vertices per line
                // Vertex 1: start position
                ConnectionVertex v1;
                v1.position = startPos;
                v1.weight = weight;
                m_connectionVertices.push_back(v1);

                // Vertex 2: end position
                ConnectionVertex v2;
                v2.position = endPos;
                v2.weight = weight;
                m_connectionVertices.push_back(v2);

                connectionCount++;
            }
        }

        std::cout << "  Layer " << layerIdx << ": " << (inputSize * outputSize)
                  << " connections\n";

        neuronOffset += inputSize;
        weightOffset += inputSize * outputSize;
    }

    m_connectionCount = connectionCount;
    std::cout << "[DEBUG] Total: " << connectionCount << " connections, "
              << m_connectionVertices.size() << " vertices\n";
}

void Renderer::uploadConnections() {
    if (m_connectionVertices.empty()) return;

    glBindVertexArray(m_connectionVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_connectionVBO);

    glBufferData(GL_ARRAY_BUFFER,
                 m_connectionVertices.size() * sizeof(ConnectionVertex),
                 m_connectionVertices.data(),
                 GL_STATIC_DRAW);

    // Vertex attribute 0: Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex),
                          (void*)offsetof(ConnectionVertex, position));

    // Vertex attribute 1: Weight
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(ConnectionVertex),
                          (void*)offsetof(ConnectionVertex, weight));

    glBindVertexArray(0);
}

void Renderer::renderConnections(const glm::mat4& viewMatrix, const glm::mat4& projMatrix) {
    if (m_connectionCount == 0) return;

    glUseProgram(m_connectionProgram);

    // Upload matrices
    GLint viewLoc = glGetUniformLocation(m_connectionProgram, "u_view");
    GLint projLoc = glGetUniformLocation(m_connectionProgram, "u_projection");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &viewMatrix[0][0]);
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, &projMatrix[0][0]);

    // Upload line width to geometry shader
    GLint lineWidthLoc = glGetUniformLocation(m_connectionProgram, "u_lineWidth");
    glUniform1f(lineWidthLoc, m_config.connectionWidth);

    // Draw lines (geometry shader will expand them into quads)
    glBindVertexArray(m_connectionVAO);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(m_connectionVertices.size()));
    glBindVertexArray(0);

    // Check for OpenGL errors
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "[ERROR] OpenGL error after connection render: " << err << "\n";
    }
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
    if (m_connectionVAO) {
        glDeleteVertexArrays(1, &m_connectionVAO);
        m_connectionVAO = 0;
    }
    if (m_connectionVBO) {
        glDeleteBuffers(1, &m_connectionVBO);
        m_connectionVBO = 0;
    }
    if (m_connectionProgram) {
        glDeleteProgram(m_connectionProgram);
        m_connectionProgram = 0;
    }
}