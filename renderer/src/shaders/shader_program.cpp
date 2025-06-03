//
// Created by brent on 11/19/24.
//

#include "shaders/shader_program.h"


#include <string>
#include <fstream>
#include <sstream>
#include <iostream>


ShaderProgram::ShaderProgram(const std::vector<ShaderInfo>& shaders) {
  _program_id = glCreateProgram(); // create program
  for (const ShaderInfo& shaderInfo: shaders) { // compile shaders
    createShader(shaderInfo);
    glAttachShader(_program_id, shaderInfo.shaderType);
  }
  glLinkProgram(_program_id); // link program

	// shaders are linked into program, so they can be deleted
	for (const ShaderInfo& shaderInfo: shaders) {
		glDeleteShader(shaderInfo.shaderType);
    }

	// check if compilation was a success
	GLint success = 0;
	glGetProgramiv(_program_id, GL_LINK_STATUS, &success);
	if (!success){
		GLchar infoLog[1024];
		glGetProgramInfoLog(_program_id, 1024, NULL, infoLog);
		std::cerr << "ERROR::SHADER::PROGRAM_LINKING_ERROR: " << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		exit(EXIT_FAILURE);
	}
}

void ShaderProgram::createShader(const ShaderInfo& shaderInfo) const {
	const std::string shaderCode = fetchShaderSrc((getShadersDir() + shaderInfo.shaderPath).c_str()); // get the shader src code
	const bool compiled = compileShader(shaderCode, shaderInfo.shaderType); // compile the shader
	if (!compiled){ // check if compilation is successful
		GLchar infoLog[1024];
		glGetShaderInfoLog(shaderInfo.shaderType, 1024, NULL, infoLog);
		std::cerr << "ERROR::SHADER::SHADER_COMPILATION_ERROR: " << getShadersDir() + shaderInfo.shaderPath << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		exit(1);
	}
}

std::string ShaderProgram::fetchShaderSrc(const char* shaderPath) const {
	std::string code;
	std::ifstream shaderFile;
	// ensure ifstream object can throw exceptions:
	shaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try{
		shaderFile.open(shaderPath); // open file
		std::stringstream shaderStream;
		shaderStream << shaderFile.rdbuf(); // read file
		shaderFile.close(); // close file
		code = shaderStream.str(); // get as string
	}
	catch (std::ifstream::failure)
	{
		std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESSFULLY_READ : " << shaderPath << std::endl;
		exit(EXIT_FAILURE);
	}
	return code;
}

bool ShaderProgram::compileShader(const std::string &shaderCode, unsigned int shaderType) const {
	const char* shaderCodeCStr = shaderCode.c_str();
	glShaderSource(shaderType, 1, &shaderCodeCStr, NULL);
	glCompileShader(shaderType);

	// check if the compilation was successful
	GLint success = 0;
	glGetShaderiv(shaderType, GL_COMPILE_STATUS, &success);
	return success;
}

std::string ShaderProgram::getShadersDir() const {
	return _cmakelistsDir + "/src/shaders/";
}

void ShaderProgram::use() const {
	glUseProgram(_program_id);
}


GLint ShaderProgram::getUniformLocation(const std::string& name) const {
	if (uniformLocationMap.find(name) != uniformLocationMap.end()) {
		return uniformLocationMap[name];
	}
	GLint location = glGetUniformLocation(_program_id, name.c_str());
	uniformLocationMap[name] = location;
	return location;
}

void ShaderProgram::setBool(const std::string& name, bool value) const{
	glUniform1i(getUniformLocation(name), (int)value);
}

void ShaderProgram::setInt(const std::string& name, int value) const{
	glUniform1i(getUniformLocation(name), value);
}

void ShaderProgram::setFloat(const std::string& name, float value) const{
	glUniform1f(getUniformLocation(name), value);
}

void ShaderProgram::setVec2(const std::string& name, const glm::vec2& value) const{
	glUniform2fv(getUniformLocation(name), 1, &value[0]);
}

void ShaderProgram::setVec2(const std::string& name, float x, float y) const{
	glUniform2f(getUniformLocation(name), x, y);
}

void ShaderProgram::setVec3(const std::string& name, const glm::vec3& value) const{
	glUniform3fv(getUniformLocation(name), 1, &value[0]);
}

void ShaderProgram::setVec3(const std::string& name, float x, float y, float z) const{
	glUniform3f(getUniformLocation(name), x, y, z);
}

void ShaderProgram::setVec4(const std::string& name, const glm::vec4& value) const{
	glUniform4fv(getUniformLocation(name), 1, &value[0]);
}

void ShaderProgram::setVec4(const std::string& name, float x, float y, float z, float w) const{
	glUniform4f(getUniformLocation(name), x, y, z, w);
}

void ShaderProgram::setMat2(const std::string& name, const glm::mat2& mat) const{
	glUniformMatrix2fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setMat3(const std::string& name, const glm::mat3& mat) const{
	glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setMat4(const std::string& name, const glm::mat4& mat) const{
	glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
}

void ShaderProgram::setIntArray(const std::string& name, const std::vector<int>& ints) const{
  glUniform1iv(getUniformLocation(name), ints.size(), ints.data());
}

void ShaderProgram::setVec2Array(const std::string& name, const std::vector<glm::vec2>& vecs) const {
	glUniform2fv(getUniformLocation(name), vecs.size(), &vecs.data()[0][0]);
}

void ShaderProgram::setVec3Array(const std::string& name, const std::vector<glm::vec3>& vecs) const {
	glUniform2fv(getUniformLocation(name), vecs.size(), &vecs.data()[0][0]);
}

void ShaderProgram::setMat4Array(const std::string& name, const std::vector<glm::mat4>& mats) const {
	glUniformMatrix4fv(getUniformLocation(name), mats.size(), GL_FALSE, &mats[0][0][0]);
}

