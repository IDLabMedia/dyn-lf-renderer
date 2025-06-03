
#ifndef DECOMPRESSOR_H
#define DECOMPRESSOR_H
#include <memory>

#include "meshes/mesh_loader.h"

class Decompressor{
protected:
  std::string _inDir;
public:
  explicit Decompressor(std::string inDir): _inDir(std::move(inDir)) {}
  virtual ~Decompressor() = default;

  /**
  * Interface to decompress anything before the shader pipeline runs for a frame.
  * Allows the updating of the mesh or texture before the shader pipeline run.
  *
  * @param frame The frame to decompress
  * @param mesh The mesh to potentially alter
  * @param texture The texture to potentially alter
  */
  virtual void decompress(size_t frame, std::unique_ptr<MeshLoader>& mesh, std::unique_ptr<TexturesLoader>& texture) = 0;
};

#endif // !DECOMPRESSOR_H
