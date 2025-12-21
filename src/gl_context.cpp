#include "gl_context.h"
#include <iostream>
#include <chrono>

GLContext::~GLContext() {
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    glfwTerminate();
}

bool GLContext::initialize(const Config& config) {
    m_config = config;

    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "[ERROR] Failed to initialize GLFW\n";
        return false;
    }

    // Configure OpenGL context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, m_config.glMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, m_config.glMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    if (m_config.enableDebugOutput) {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    }

    // Create window
    m_window = glfwCreateWindow(m_config.width, m_config.height,
                                 m_config.title.c_str(), nullptr, nullptr);
    if (!m_window) {
        std::cerr << "[ERROR] Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);

    // Set VSync
    glfwSwapInterval(m_config.enableVSync ? 1 : 0);

    // Load OpenGL function pointers with GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[ERROR] Failed to initialize GLAD\n";
        glfwDestroyWindow(m_window);
        glfwTerminate();
        return false;
    }

    // Setup callbacks
    setupCallbacks();

    // Enable OpenGL debug output
    if (m_config.enableDebugOutput) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(debugMessageCallback, nullptr);

        // Filter out low severity messages
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION,
                              0, nullptr, GL_FALSE);
    }

    // Print system info
    printSystemInfo();

    return true;
}

void GLContext::run(const std::function<void(float)>& updateCallback,
                    const std::function<void()>& renderCallback) const {
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(m_window)) {
        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        const float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        // Poll events
        glfwPollEvents();

        // Update
        if (updateCallback) {
            updateCallback(deltaTime);
        }

        // Render
        if (renderCallback) {
            renderCallback();
        }

        // Swap buffers
        glfwSwapBuffers(m_window);
    }
}

void GLContext::requestClose() {
    if (m_window) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
}

void GLContext::getFramebufferSize(int& width, int& height) const {
    glfwGetFramebufferSize(m_window, &width, &height);
}

bool GLContext::shouldClose() const {
    return glfwWindowShouldClose(m_window);
}

void GLContext::setupCallbacks() {
    // Store 'this' pointer for callback access
    glfwSetWindowUserPointer(m_window, this);

    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    glfwSetKeyCallback(m_window, keyCallback);
}

void GLContext::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void GLContext::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)scancode;  // Unused
    (void)mods;      // Unused

    if (action == GLFW_PRESS) {
        // ESC to close
        if (key == GLFW_KEY_ESCAPE) {
            auto* context = static_cast<GLContext*>(glfwGetWindowUserPointer(window));
            if (context) {
                context->requestClose();
            }
        }
    }
}

void GLAPIENTRY GLContext::debugMessageCallback(const GLenum source, const GLenum type, const GLuint id,
                                                 const GLenum severity, const GLsizei length,
                                                 const GLchar* message, const void* userParam) {
    (void)source;
    (void)length;
    (void)userParam;

    // Ignore non-significant error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    std::cerr << "---------------\n";
    std::cerr << "OpenGL Debug Message (" << id << "): " << message << "\n";

    switch (source) {
        case GL_DEBUG_SOURCE_API:             std::cerr << "Source: API"; break;
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cerr << "Source: Window System"; break;
        case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cerr << "Source: Shader Compiler"; break;
        case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cerr << "Source: Third Party"; break;
        case GL_DEBUG_SOURCE_APPLICATION:     std::cerr << "Source: Application"; break;
        case GL_DEBUG_SOURCE_OTHER:           std::cerr << "Source: Other"; break;
        default: ;
    }
    std::cerr << "\n";

    switch (type) {
        case GL_DEBUG_TYPE_ERROR:               std::cerr << "Type: Error"; break;
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cerr << "Type: Deprecated Behaviour"; break;
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cerr << "Type: Undefined Behaviour"; break;
        case GL_DEBUG_TYPE_PORTABILITY:         std::cerr << "Type: Portability"; break;
        case GL_DEBUG_TYPE_PERFORMANCE:         std::cerr << "Type: Performance"; break;
        case GL_DEBUG_TYPE_MARKER:              std::cerr << "Type: Marker"; break;
        case GL_DEBUG_TYPE_PUSH_GROUP:          std::cerr << "Type: Push Group"; break;
        case GL_DEBUG_TYPE_POP_GROUP:           std::cerr << "Type: Pop Group"; break;
        case GL_DEBUG_TYPE_OTHER:               std::cerr << "Type: Other"; break;
        default: ;
    }
    std::cerr << "\n";

    switch (severity) {
        case GL_DEBUG_SEVERITY_HIGH:         std::cerr << "Severity: high"; break;
        case GL_DEBUG_SEVERITY_MEDIUM:       std::cerr << "Severity: medium"; break;
        case GL_DEBUG_SEVERITY_LOW:          std::cerr << "Severity: low"; break;
        case GL_DEBUG_SEVERITY_NOTIFICATION: std::cerr << "Severity: notification"; break;
        default: ;
    }
    std::cerr << "\n\n";
}

void GLContext::printSystemInfo() {
    std::cout << "========================================\n";
    std::cout << "OpenGL System Information\n";
    std::cout << "========================================\n";
    std::cout << "Vendor:   " << glGetString(GL_VENDOR) << "\n";
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << "\n";
    std::cout << "Version:  " << glGetString(GL_VERSION) << "\n";
    std::cout << "GLSL:     " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";

    // Check compute shader support
    GLint maxComputeWorkGroupCount[3];
    GLint maxComputeWorkGroupSize[3];
    GLint maxComputeWorkGroupInvocations;

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &maxComputeWorkGroupCount[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 1, &maxComputeWorkGroupCount[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 2, &maxComputeWorkGroupCount[2]);

    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxComputeWorkGroupSize[0]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxComputeWorkGroupSize[1]);
    glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 2, &maxComputeWorkGroupSize[2]);

    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &maxComputeWorkGroupInvocations);

    std::cout << "\nCompute Shader Capabilities:\n";
    std::cout << "  Max Work Group Count: ("
              << maxComputeWorkGroupCount[0] << ", "
              << maxComputeWorkGroupCount[1] << ", "
              << maxComputeWorkGroupCount[2] << ")\n";
    std::cout << "  Max Work Group Size:  ("
              << maxComputeWorkGroupSize[0] << ", "
              << maxComputeWorkGroupSize[1] << ", "
              << maxComputeWorkGroupSize[2] << ")\n";
    std::cout << "  Max Invocations:      " << maxComputeWorkGroupInvocations << "\n";
    std::cout << "========================================\n";
}