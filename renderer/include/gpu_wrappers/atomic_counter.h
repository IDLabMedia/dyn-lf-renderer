//
// Created by brent on 3/12/25.
//

#ifndef ATOMIC_COUNTER_H
#define ATOMIC_COUNTER_H

#include <glad/glad.h>

class AtomicCounter {
private:
    GLuint _bufferId = 0;

public:
    AtomicCounter();

    ~AtomicCounter();

    /**
     * Binds the atomic counter buffer to a binding point.
     * Must match layout(binding=X) in the shader.
     */
    void bind(GLuint bindingPoint) const;

    /**
     * Set the atomic value.
     */
    void set(GLuint value) const;

    /**
     * Reads back the current value of the atomic counter.
     */
    GLuint get() const;

    GLuint getBufferId() const;
};
#endif //ATOMIC_COUNTER_H
