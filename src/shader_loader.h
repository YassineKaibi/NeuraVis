#pragma once

#include <glad/glad.h>
#include <string>
#include <vector>

/**
 * @brief Utility class for loading and compiling GLSL shaders
 *
 * Supports:
 * - Vertex, Fragment, Geometry, Compute shaders
 * - Detailed error reporting with line numbers
 * - Shader program linking
 * - Hot reload capability (Phase 2)
 */
class ShaderLoader {
public:
    /**
     * @brief Load and compile a compute shader from file
     * @param filepath Path to .comp file
     * @return Compiled shader program ID, or 0 on failure
     */
    static GLuint loadComputeShader(const std::string& filepath);

    /**
     * @brief Load and compile vertex + fragment shaders from files
     * @param vertPath Path to .vert file
     * @param fragPath Path to .frag file
     * @return Linked shader program ID, or 0 on failure
     */
    static GLuint loadShaderProgram(const std::string& vertPath,
                                     const std::string& fragPath);

    /**
     * @brief Load and compile vertex + geometry + fragment shaders
     */
    static GLuint loadShaderProgram(const std::string& vertPath,
                                     const std::string& geomPath,
                                     const std::string& fragPath);

    /**
     * @brief Read shader source code from file
     * @param filepath Path to shader file
     * @param outSource Output string to store source code
     * @return true if file read successfully
     */
    static bool readShaderFile(const std::string& filepath, std::string& outSource);

    /**
     * @brief Compile a shader from source string
     * @param shaderType GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPUTE_SHADER, etc.
     * @param source Shader source code string
     * @return Compiled shader ID, or 0 on failure
     */
    static GLuint compileShader(GLenum shaderType, const std::string& source);

    /**
     * @brief Link multiple compiled shaders into a program
     * @param shaders Vector of compiled shader IDs
     * @return Linked program ID, or 0 on failure
     */
    static GLuint linkProgram(const std::vector<GLuint>& shaders);

    /**
     * @brief Check shader compilation status and print errors
     * @param shader Shader ID to check
     * @param filepath Optional filepath for better error messages
     * @return true if compilation succeeded
     */
    static bool checkCompileErrors(GLuint shader, const std::string& filepath = "");

    /**
     * @brief Check program linking status and print errors
     * @param program Program ID to check
     * @return true if linking succeeded
     */
    static bool checkLinkErrors(GLuint program);

private:
    static std::string shaderTypeToString(GLenum shaderType);
};