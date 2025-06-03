//
// Created by brent on 12/10/24.
//

#ifndef JSON_READER_H
#define JSON_READER_H
#include <nlohmann/json.hpp>

/**
 * Read a json file to a json object
 *
 * @param jsonPath The path of the json file.
 * @return The json object
 */
nlohmann::json readJsonFile(const std::string& jsonPath);

#endif //JSON_READER_H
