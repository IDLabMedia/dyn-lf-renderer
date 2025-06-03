/*
* Created by brent on 19/05/2025
*/

#pragma once

#include "gpu_wrappers/atomic_counter.h"
#include <glad/glad.h>

struct IndirectDrawCommandData {
  GLuint count;
  GLuint instanceCount;
  GLuint firstIndex;
  GLuint baseVertex;
  GLuint baseInstance;
};

class IndirectDrawCommand{
private:
  GLuint _bufferId = 0;
  IndirectDrawCommandData _command =  {
    .count = 0,
    .instanceCount = 1,
    .firstIndex = 0,
    .baseVertex = 0,
    .baseInstance = 0,
  };
  
public:
  IndirectDrawCommand();
  ~IndirectDrawCommand();

  void updateCount(const AtomicCounter& counter);

  GLuint getBufferId() const;
};
