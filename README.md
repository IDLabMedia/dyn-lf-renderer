# Real-Time dynamic light field renderer

This real-time dynamic light field renderer can generate novel viewpoints with 6 degrees of freedom at 90+ FPS from a multi-view video dataset. It enables immersive, high-performance view synthesis for interactive applications.

## Project structure

The system consists of two main components:

- [**Preprocessor**](./preprocessor/README.md): Offline transformation of a multi-view dataset into an intermediate representation used by the renderer.
- [**Renderer**](./renderer/README.md): Real-time renderer using OpenGL to synthesize novel views from the intermediate representation.
