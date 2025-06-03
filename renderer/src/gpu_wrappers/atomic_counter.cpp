//
// Created by brent on 3/12/25.
//

#include "gpu_wrappers/atomic_counter.h"
#include "profiler.h"

AtomicCounter::AtomicCounter() {
    glGenBuffers(1, &_bufferId);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, _bufferId);
    GLuint zero = 0;
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), &zero, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
}

AtomicCounter::~AtomicCounter() {
    glDeleteBuffers(1, &_bufferId);
}

void AtomicCounter::bind(const GLuint bindingPoint) const {
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, bindingPoint, _bufferId);
}

void AtomicCounter::set(const GLuint value) const {
    // glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, _bufferId);
    // glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &value);
    // glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

    // non blocking write counter
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, _bufferId);
    GLuint* ptr = (GLuint*)glMapBufferRange(
        GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint),
        GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT
    );
    if (ptr) {
        *ptr = value;
        glUnmapBuffer(GL_ATOMIC_COUNTER_BUFFER);
    }
}


GLuint AtomicCounter::get() const {
    PROFILE_SCOPE("counter_get");
    GLuint value = 0;
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, _bufferId);
    glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &value);
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
    return value;
}


GLuint AtomicCounter::getBufferId() const {
    return _bufferId;
}

