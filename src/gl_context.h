#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>
#include <functional>

/**
 * Manages OpenGL context creation, window management, and debug output
 *
 * Responsibilities:
 * - GLFW initialization and window creation
 * - OpenGL 4.6 Core context setup
 * - Debug message callback registration
 * - Input handling callbacks
 * - Main render loop
 */

class GLContext {
public:
    struct Config {
        int width = 1280;
        int height = 720;
        std::string title = "GPU Neural Network Visualizer";
        bool enableVSync = true;
        bool enableDebugOutput = true;
        int glMajorVersion = 4;
        int glMinorVersion = 6;
    };

    GLContext() = default;
    ~GLContext();

    // Prevent copying
    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;

    /**
     * @brief Initialize GLFW, create window, and setup OpenGL context
     * @param config Window and OpenGL configuration
     * @return true if initialization successful
     */
    bool initialize(const Config& config);

    /**
     * @brief Start the main render loop
     * @param updateCallback Called each frame for logic updates
     * @param renderCallback Called each frame for rendering
     */
    void run(const std::function<void(float deltaTime)>& updateCallback,
             const std::function<void()>& renderCallback) const;

    /**
     * @brief Request window to close (e.g., from ESC key)
     */
    void requestClose();

    /**
     * @brief Get the GLFW window handle
     */
    GLFWwindow* getWindow() const { return m_window; }

    /**
     * @brief Get current framebuffer size (for viewport)
     */
    void getFramebufferSize(int& width, int& height) const;

    /**
     * @brief Check if window should close
     */
    bool shouldClose() const;

private:
    GLFWwindow* m_window = nullptr;
    Config m_config;

    // Callbacks
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void GLAPIENTRY debugMessageCallback(GLenum source, GLenum type, GLuint id,
                                                 GLenum severity, GLsizei length,
                                                 const GLchar* message, const void* userParam);

    void setupCallbacks();

    static void printSystemInfo();
};