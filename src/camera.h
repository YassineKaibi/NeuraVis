#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

/**
 * @brief Simple orbital camera for viewing the neural network
 */
class Camera {
public:
    Camera() {
        updateViewMatrix();
    }

    /**
     * @brief Get view matrix
     */
    glm::mat4 getViewMatrix() const {
        return m_viewMatrix;
    }

    /**
     * @brief Get projection matrix
     */
    glm::mat4 getProjectionMatrix(float aspectRatio) const {
        return glm::perspective(glm::radians(m_fov), aspectRatio, m_near, m_far);
    }

    /**
     * @brief Orbit around target
     */
    void orbit(float deltaYaw, float deltaPitch) {
        m_yaw += deltaYaw;
        m_pitch += deltaPitch;

        // Clamp pitch to avoid gimbal lock
        m_pitch = glm::clamp(m_pitch, -89.0f, 89.0f);

        updateViewMatrix();
    }

    /**
     * @brief Zoom in/out
     */
    void zoom(float delta) {
        m_distance += delta;
        m_distance = glm::clamp(m_distance, 1.0f, 50.0f);
        updateViewMatrix();
    }

    /**
     * @brief Set camera target (what we're looking at)
     */
    void setTarget(const glm::vec3& target) {
        m_target = target;
        updateViewMatrix();
    }

private:
    glm::vec3 m_target = glm::vec3(0.0f, 0.0f, 0.0f);
    float m_distance = 10.0f;
    float m_yaw = 0.0f;      // Rotation around Y axis
    float m_pitch = 0.0f;    // Rotation around X axis

    float m_fov = 45.0f;
    float m_near = 0.1f;
    float m_far = 100.0f;

    glm::mat4 m_viewMatrix;

    void updateViewMatrix() {
        // Calculate camera position from spherical coordinates
        float x = m_distance * cos(glm::radians(m_pitch)) * cos(glm::radians(m_yaw));
        float y = m_distance * sin(glm::radians(m_pitch));
        float z = m_distance * cos(glm::radians(m_pitch)) * sin(glm::radians(m_yaw));

        glm::vec3 position = m_target + glm::vec3(x, y, z);

        m_viewMatrix = glm::lookAt(position, m_target, glm::vec3(0.0f, 1.0f, 0.0f));
    }
};