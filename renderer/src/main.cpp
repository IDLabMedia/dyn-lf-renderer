#include <glad/glad.h>
#include <GLFW/glfw3.h> // needs to be after glad.h
#define GLM_ENABLE_EXPERIMENTAL false

#include "application.h"
#include "profiler.h"

int main(int argc, char* argv[]) {

#if ENABLE_PROFILING
  Profiler::get().beginSession("../../plots/data/time/frog_vr.json");
#endif

    // parse arguments
    ProgramInfo info = ProgramInfo(argc, argv);

    // create the application and init it
    Application app;
    app.initialize(info);

	// app class will automatically create a window
	// destructor of app class will safely shut down
	// the window, textures, shaders etc.

	app.run(); // run the main loop of the program

#if ENABLE_PROFILING
    Profiler::get().endSession();
#endif

	return 0;
}
