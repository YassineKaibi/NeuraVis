#include "gl_context.h"
#include <iostream>

/**
 * GPU-Based Neural Network Visualizer
 *
 * Phase 0: Environment Setup & Verification
 *
 * This minimal version:
 * 1. Creates OpenGL 4.6 context
 * 2. Verifies GPU capabilities
 * 3. Prints system information
 * 4. Clears screen to a color (proof of rendering)
 */

int main() {
    std::cout << "===========================================\n";
    std::cout << "GPU Neural Network Visualizer - Setup Test\n";
    std::cout << "===========================================\n\n";

    // Initialize OpenGL context
    GLContext context;
    GLContext::Config config;
    config.width = 1280;
    config.height = 720;
    config.title = "NN Visualizer - Setup Test";
    config.enableDebugOutput = true;

    if (!context.initialize(config)) {
        std::cerr << "[ERROR] Failed to initialize OpenGL context\n";
        return -1;
    }

    std::cout << "\n[SUCCESS] OpenGL context created successfully!\n";
    std::cout << "Press ESC to close window...\n\n";

    // Simple render loop - clear to a color
    float hue = 0.0f;
    context.run(
        // Update callback
        [&](float deltaTime) {
            hue += deltaTime * 0.1f;  // Slowly cycle hue
            if (hue > 1.0f) hue -= 1.0f;
        },
        // Render callback
        [&]() {
            // HSV to RGB conversion (simple approximation)
            float r = std::abs(hue * 6.0f - 3.0f) - 1.0f;
            float g = 2.0f - std::abs(hue * 6.0f - 2.0f);
            float b = 2.0f - std::abs(hue * 6.0f - 4.0f);
            r = std::clamp(r, 0.0f, 1.0f);
            g = std::clamp(g, 0.0f, 1.0f);
            b = std::clamp(b, 0.0f, 1.0f);

            glClearColor(r * 0.3f, g * 0.3f, b * 0.3f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
        }
    );

    std::cout << "\n[INFO] Application closed normally\n";
    return 0;
}