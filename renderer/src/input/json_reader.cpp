//
// Created by brent on 12/10/24.
//
#include "input/json_reader.h"

#include <fstream>
#include <iostream>


nlohmann::json readJsonFile(const std::string& jsonPath) {
	// open the file
	std::ifstream file;
	file.open(jsonPath);
	if(!file.is_open()) {
		std::cerr << "ERROR::JSON::FAILED_TO_OPEN : " << jsonPath << std::endl;
		exit(EXIT_FAILURE);
	}

	// read and close file
	nlohmann::json outJson;
	try {
		file >> outJson;
	}catch (nlohmann::json::parse_error &e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "ERROR::JSON::FAILED_TO_PARSE : " << jsonPath << std::endl;
		file.close();
		exit(EXIT_FAILURE);
	}
	file.close();

	return outJson;
}
