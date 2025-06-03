/*
* Created by brent on 19/05/2025
*/

#include "gpu_wrappers/indirect_command.h"
#include "gpu_wrappers/atomic_counter.h"


IndirectDrawCommand::IndirectDrawCommand() {
    glGenBuffers(1, &_bufferId);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, _bufferId);
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(_command), &_command, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);
}

IndirectDrawCommand::~IndirectDrawCommand(){
  glDeleteBuffers(1, &_bufferId);
}

void IndirectDrawCommand::updateCount(const AtomicCounter& counter){
  glCopyNamedBufferSubData(counter.getBufferId(), _bufferId, 0, 0, sizeof(GLuint));
}

GLuint IndirectDrawCommand::getBufferId() const{
    return _bufferId;
}


