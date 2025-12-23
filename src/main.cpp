#include "gl_context.h"
#include "nn_buffers.h"
#include "nn_compute.h"
#include "renderer.h"
#include "camera.h"
#include <iostream>
#include <vector>

/**
 * GPU-Based Neural Network Visualizer
 *
 * Features:
 * - Real-time XOR network visualization
 * - GPU-accelerated forward propagation
 * - Interactive camera controls
 * - Color-coded neuron activations
 */

// Global state for mouse input
struct MouseState {
    double lastX = 0.0;
    double lastY = 0.0;
    bool firstMouse = true;
    bool leftButtonPressed = false;
    bool rightButtonPressed = false;
};

MouseState g_mouse;
Camera* g_camera = nullptr;

// Mouse callbacks
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_mouse.leftButtonPressed = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        g_mouse.rightButtonPressed = (action == GLFW_PRESS);
    }

    if (action == GLFW_PRESS) {
        g_mouse.firstMouse = true;  // Reset on button press
    }
}

void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos) {
    if (g_mouse.firstMouse) {
        g_mouse.lastX = xpos;
        g_mouse.lastY = ypos;
        g_mouse.firstMouse = false;
        return;
    }

    float deltaX = static_cast<float>(xpos - g_mouse.lastX);
    float deltaY = static_cast<float>(ypos - g_mouse.lastY);

    g_mouse.lastX = xpos;
    g_mouse.lastY = ypos;

    if (g_camera) {
        if (g_mouse.leftButtonPressed) {
            // Orbit camera
            g_camera->orbit(deltaX * 0.5f, -deltaY * 0.5f);
        }
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    if (g_camera) {
        g_camera->zoom(-static_cast<float>(yoffset) * 0.5f);
    }
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "GPU Neural Network Visualizer - XOR Demo\n";
    std::cout << "===========================================\n\n";

    // ========================================
    // 1. Initialize OpenGL Context
    // ========================================
    GLContext context;
    GLContext::Config config;
    config.width = 1280;
    config.height = 720;
    config.title = "NeuraVis - XOR Network";
    config.enableDebugOutput = true;

    if (!context.initialize(config)) {
        std::cerr << "[ERROR] Failed to initialize OpenGL context\n";
        return -1;
    }

    // Set up mouse callbacks
    glfwSetMouseButtonCallback(context.getWindow(), mouseButtonCallback);
    glfwSetCursorPosCallback(context.getWindow(), mouseMoveCallback);
    glfwSetScrollCallback(context.getWindow(), scrollCallback);

    // ========================================
    // 2. Create XOR Neural Network
    // ========================================
    std::cout << "\n[INFO] Setting up XOR network (2 -> 2 -> 1)\n";

    // Network topology: 2 inputs -> 2 hidden -> 1 output
    std::vector<uint32_t> topology = {2, 2, 1};
    std::vector<uint32_t> activations = {0, 0};  // ReLU for all layers

    NeuralBuffers buffers;
    buffers.initialize(topology, activations);

    // XOR weights (hand-crafted for demonstration)
    // Layer 0: 2x2 = 4 weights
    // Layer 1: 2x1 = 2 weights
    std::vector<float> weights = {
        // Layer 0 weights (2 inputs -> 2 hidden neurons)
        1.0f, 1.0f,    // Hidden neuron 0: w0, w1
        1.0f, 1.0f,    // Hidden neuron 1: w0, w1

        // Layer 1 weights (2 hidden -> 1 output neuron)
        1.0f, -2.0f    // Output neuron: w0, w1
    };

    // XOR biases
    // Layer 0: 2 biases (one per hidden neuron)
    // Layer 1: 1 bias (one for output neuron)
    std::vector<float> biases = {
        0.0f, -1.5f,   // Layer 0 biases
        0.0f           // Layer 1 bias
    };

    buffers.uploadWeights(weights);
    buffers.uploadBiases(biases);

    std::cout << "[INFO] Network initialized with " << weights.size()
              << " weights and " << biases.size() << " biases\n";

    // ========================================
    // 3. Initialize Compute Shader
    // ========================================
    NeuralCompute compute;
    if (!compute.initialize("shaders/forward.comp", buffers)) {
        std::cerr << "[ERROR] Failed to initialize compute shader\n";
        return -1;
    }

    // ========================================
    // 4. Initialize Renderer
    // ========================================
    Renderer renderer;
    if (!renderer.initialize(buffers)) {
        std::cerr << "[ERROR] Failed to initialize renderer\n";
        return -1;
    }

    // ========================================
    // 5. Set up Camera
    // ========================================
    Camera camera;
    g_camera = &camera;

    // Center camera on network (middle layer)
    camera.setTarget(glm::vec3(3.0f, 0.0f, 0.0f));
    camera.zoom(0.0f);  // Set initial distance

    // ========================================
    // 6. OpenGL State Setup
    // ========================================
    glEnable(GL_DEPTH_TEST);   // Re-enabled now that positioning is fixed
    glEnable(GL_PROGRAM_POINT_SIZE);  // Allow shaders to set point size
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ========================================
    // 7. Test XOR Network
    // ========================================
    std::cout << "\n[INFO] Testing XOR network:\n";

    struct XORTest {
        std::vector<float> input;
        std::string label;
    };

    std::vector<XORTest> tests = {
        {{0.0f, 0.0f}, "(0,0) -> 0"},
        {{0.0f, 1.0f}, "(0,1) -> 1"},
        {{1.0f, 0.0f}, "(1,0) -> 1"},
        {{1.0f, 1.0f}, "(1,1) -> 0"}
    };

    int currentTest = 1;  // Start with (0,1) which has non-zero output

    // Set initial input
    buffers.setInputs(tests[currentTest].input);
    compute.forward();

    // Print initial results
    std::vector<float> outputs;
    buffers.readOutputs(outputs);
    std::cout << "  " << tests[currentTest].label << " = " << outputs[0] << "\n";

    std::cout << "\n[INFO] Controls:\n";
    std::cout << "  Mouse Left Drag: Rotate camera\n";
    std::cout << "  Mouse Scroll: Zoom\n";
    std::cout << "  1-4: Test XOR inputs (0,0) (0,1) (1,0) (1,1)\n";
    std::cout << "  SPACE: Run forward pass\n";
    std::cout << "  C: Toggle connection visualization (currently OFF)\n";
    std::cout << "  ESC: Exit\n\n";

    // ========================================
    // 8. Main Render Loop
    // ========================================
    bool needsForwardPass = false;

    context.run(
        // Update callback
        [&](float deltaTime) {
            // Handle keyboard input for XOR tests
            static bool key1WasPressed = false, key2WasPressed = false;
            static bool key3WasPressed = false, key4WasPressed = false;

            bool key1Pressed = glfwGetKey(context.getWindow(), GLFW_KEY_1) == GLFW_PRESS;
            bool key2Pressed = glfwGetKey(context.getWindow(), GLFW_KEY_2) == GLFW_PRESS;
            bool key3Pressed = glfwGetKey(context.getWindow(), GLFW_KEY_3) == GLFW_PRESS;
            bool key4Pressed = glfwGetKey(context.getWindow(), GLFW_KEY_4) == GLFW_PRESS;

            if (key1Pressed && !key1WasPressed) {
                currentTest = 0;
                buffers.setInputs(tests[currentTest].input);
                needsForwardPass = true;
                std::cout << "[TEST] Input: " << tests[currentTest].label << "\n";
            }
            if (key2Pressed && !key2WasPressed) {
                currentTest = 1;
                buffers.setInputs(tests[currentTest].input);
                needsForwardPass = true;
                std::cout << "[TEST] Input: " << tests[currentTest].label << "\n";
            }
            if (key3Pressed && !key3WasPressed) {
                currentTest = 2;
                buffers.setInputs(tests[currentTest].input);
                needsForwardPass = true;
                std::cout << "[TEST] Input: " << tests[currentTest].label << "\n";
            }
            if (key4Pressed && !key4WasPressed) {
                currentTest = 3;
                buffers.setInputs(tests[currentTest].input);
                needsForwardPass = true;
                std::cout << "[TEST] Input: " << tests[currentTest].label << "\n";
            }

            key1WasPressed = key1Pressed;
            key2WasPressed = key2Pressed;
            key3WasPressed = key3Pressed;
            key4WasPressed = key4Pressed;

            // Manual forward pass trigger
            static bool spaceWasPressed = false;
            bool spacePressed = glfwGetKey(context.getWindow(), GLFW_KEY_SPACE) == GLFW_PRESS;
            if (spacePressed && !spaceWasPressed) {
                needsForwardPass = true;
                std::cout << "[INFO] Running forward pass...\n";
            }
            spaceWasPressed = spacePressed;

            // Toggle connection visualization
            static bool cWasPressed = false;
            bool cPressed = glfwGetKey(context.getWindow(), GLFW_KEY_C) == GLFW_PRESS;
            if (cPressed && !cWasPressed) {
                auto config = renderer.getConfig();
                config.showConnections = !config.showConnections;
                renderer.setConfig(config);
                std::cout << "[INFO] Connections: " << (config.showConnections ? "ON" : "OFF") << "\n";
            }
            cWasPressed = cPressed;

            // Run forward pass if needed
            if (needsForwardPass) {
                compute.forward();

                // Read and print all activations for debugging
                std::vector<float> allActivations;
                buffers.readAllActivations(allActivations);

                std::cout << "  Activations: ";
                for (size_t i = 0; i < allActivations.size(); ++i) {
                    std::cout << allActivations[i];
                    if (i < allActivations.size() - 1) std::cout << ", ";
                }
                std::cout << "\n";

                std::cout << "  Output: " << allActivations.back() << "\n\n";

                needsForwardPass = false;
            }
        },

        // Render callback
        [&]() {
            // Clear screen
            glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // Get viewport dimensions for aspect ratio
            int width, height;
            context.getFramebufferSize(width, height);
            float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
            glViewport(0, 0, width, height);

            // Get camera matrices
            glm::mat4 viewMatrix = camera.getViewMatrix();
            glm::mat4 projMatrix = camera.getProjectionMatrix(aspectRatio);

            // Render neural network
            renderer.render(viewMatrix, projMatrix);
        }
    );

    std::cout << "\n[INFO] Application closed normally\n";
    return 0;
}
