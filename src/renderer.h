#pragma once

#include "nn_buffers.h"
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>

/**
 * @brief Renders neural network visualization
 *
 *  Features:
 * - Neurons as instanced point sprites
 * - Color-coded by activation value
 * - Size proportional to activation magnitude
 * - Perceptually uniform colormap (viridis)
 */
class Renderer {
public:
    struct VisualizationConfig {
        float neuronSize = 0.3f;       // Increased from 0.1 for better visibility
        bool useViridisColormap = true;
        bool showConnections = false;  // Start with connections OFF (press C to enable)
        float minActivation = -1.0f;
        float maxActivation = 1.0f;
        float connectionAlpha = 0.6f;  // Connection transparency
    };

    Renderer() = default;
    ~Renderer();

    // Prevent copying
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /**
     * @brief Initialize rendering resources
     * @param buffers Reference to neural network buffers
     * @return true if initialization successful
     */
    bool initialize(NeuralBuffers& buffers);

    /**
     * @brief Render the neural network visualization
     * @param viewMatrix Camera view matrix
     * @param projMatrix Camera projection matrix
     */
    void render(const glm::mat4& viewMatrix, const glm::mat4& projMatrix);

    /**
     * @brief Update visualization configuration
     */
    void setConfig(const VisualizationConfig& config) { m_config = config; }

    /**
     * @brief Get current configuration
     */
    const VisualizationConfig& getConfig() const { return m_config; }

private:
    GLuint m_neuronVAO = 0;
    GLuint m_neuronPositionVBO = 0;     // Per-neuron positions (3D layout)
    GLuint m_neuronProgram = 0;         // Vertex + Fragment shader

    GLuint m_connectionVAO = 0;
    GLuint m_connectionVBO = 0;         // Connection line data
    GLuint m_connectionProgram = 0;     // Connection shader program
    uint32_t m_connectionCount = 0;

    NeuralBuffers* m_buffers = nullptr;
    VisualizationConfig m_config;

    std::vector<glm::vec3> m_neuronPositions;  // 3D positions for visualization
    uint32_t m_totalNeurons = 0;

    struct ConnectionVertex {
        glm::vec3 position;  // Either start or end position
        float weight;
    };
    std::vector<ConnectionVertex> m_connectionVertices;

    void generateNeuronLayout();
    void uploadNeuronPositions();
    void generateConnections();
    void uploadConnections();
    void renderConnections(const glm::mat4& viewMatrix, const glm::mat4& projMatrix);
    void cleanup();
};