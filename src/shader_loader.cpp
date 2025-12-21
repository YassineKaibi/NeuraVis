#include "shader_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>

GLuint ShaderLoader::loadComputeShader(const std::string& filepath) {
    std::string source;
    if (!readShaderFile(filepath, source)) {
        return 0;
    }

    GLuint shader = compileShader(GL_COMPUTE_SHADER, source);
    if (shader == 0) {
        return 0;
    }

    GLuint program = linkProgram({shader});
    glDeleteShader(shader);  // Cleanup individual shader after linking

    return program;
}

GLuint ShaderLoader::loadShaderProgram(const std::string& vertPath,
                                        const std::string& fragPath) {
    std::string vertSource, fragSource;

    if (!readShaderFile(vertPath, vertSource) || !readShaderFile(fragPath, fragSource)) {
        return 0;
    }

    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertSource);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragSource);

    if (vertShader == 0 || fragShader == 0) {
        if (vertShader) glDeleteShader(vertShader);
        if (fragShader) glDeleteShader(fragShader);
        return 0;
    }

    GLuint program = linkProgram({vertShader, fragShader});

    glDeleteShader(vertShader);
    glDeleteShader(fragShader);

    return program;
}

GLuint ShaderLoader::loadShaderProgram(const std::string& vertPath,
                                        const std::string& geomPath,
                                        const std::string& fragPath) {
    std::string vertSource, geomSource, fragSource;

    if (!readShaderFile(vertPath, vertSource) ||
        !readShaderFile(geomPath, geomSource) ||
        !readShaderFile(fragPath, fragSource)) {
        return 0;
    }

    GLuint vertShader = compileShader(GL_VERTEX_SHADER, vertSource);
    GLuint geomShader = compileShader(GL_GEOMETRY_SHADER, geomSource);
    GLuint fragShader = compileShader(GL_FRAGMENT_SHADER, fragSource);

    if (vertShader == 0 || geomShader == 0 || fragShader == 0) {
        if (vertShader) glDeleteShader(vertShader);
        if (geomShader) glDeleteShader(geomShader);
        if (fragShader) glDeleteShader(fragShader);
        return 0;
    }

    GLuint program = linkProgram({vertShader, geomShader, fragShader});

    glDeleteShader(vertShader);
    glDeleteShader(geomShader);
    glDeleteShader(fragShader);

    return program;
}

bool ShaderLoader::readShaderFile(const std::string& filepath, std::string& outSource) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Failed to open shader file: " << filepath << "\n";
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    outSource = buffer.str();

    return true;
}

GLuint ShaderLoader::compileShader(GLenum shaderType, const std::string& source) {
    GLuint shader = glCreateShader(shaderType);

    const char* sourceCStr = source.c_str();
    glShaderSource(shader, 1, &sourceCStr, nullptr);
    glCompileShader(shader);

    if (!checkCompileErrors(shader, shaderTypeToString(shaderType))) {
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

GLuint ShaderLoader::linkProgram(const std::vector<GLuint>& shaders) {
    GLuint program = glCreateProgram();

    for (GLuint shader : shaders) {
        glAttachShader(program, shader);
    }

    glLinkProgram(program);

    if (!checkLinkErrors(program)) {
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

bool ShaderLoader::checkCompileErrors(GLuint shader, const std::string& filepath) {
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);

    if (!success) {
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);

        std::vector<char> infoLog(logLength);
        glGetShaderInfoLog(shader, logLength, nullptr, infoLog.data());

        std::cerr << "[ERROR] Shader compilation failed";
        if (!filepath.empty()) {
            std::cerr << " (" << filepath << ")";
        }
        std::cerr << ":\n" << infoLog.data() << "\n";

        return false;
    }

    return true;
}

bool ShaderLoader::checkLinkErrors(GLuint program) {
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);

    if (!success) {
        GLint logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);

        std::vector<char> infoLog(logLength);
        glGetProgramInfoLog(program, logLength, nullptr, infoLog.data());

        std::cerr << "[ERROR] Program linking failed:\n" << infoLog.data() << "\n";

        return false;
    }

    return true;
}

std::string ShaderLoader::shaderTypeToString(GLenum shaderType) {
    switch (shaderType) {
        case GL_VERTEX_SHADER:   return "Vertex Shader";
        case GL_FRAGMENT_SHADER: return "Fragment Shader";
        case GL_GEOMETRY_SHADER: return "Geometry Shader";
        case GL_COMPUTE_SHADER:  return "Compute Shader";
        default:                 return "Unknown Shader";
    }
}