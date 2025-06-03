/**
* Created by Brent Matthys on 06/04/2025
*/

#include "codec/structure/frame.h"
#include "codec/io/yuv_reader.h"
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

Frame::Frame(size_t width, size_t height, size_t blocksize):
  _rows(height), _cols(width), _blockSize(blocksize)
{
  size_t lumaSize = _rows*_cols;
  size_t chromaSize = lumaSize/4; // 4:2:0 subsampling

  _Y = std::vector<int>(lumaSize);
  _U = std::vector<int>(chromaSize);
  _V = std::vector<int>(chromaSize);

  initBlockStructure();
}

Frame::Frame(const std::string& path, size_t frame, size_t width, size_t height, size_t blocksize):
  _rows(height), _cols(width), _blockSize(blocksize)
{
  // get the data
  std::vector<int> data = readRawYUVFrame(path, frame, width, height);

  // create the buffers for the frame
  size_t lumaSize = _rows*_cols;
  size_t chromaSize = lumaSize/4; // 4:2:0 subsampling

  // move the data chunks to the buffers
  _Y = std::vector<int>(data.begin(), data.begin() + lumaSize);
  _U = std::vector<int>(data.begin() + lumaSize, data.begin() + lumaSize + chromaSize);
  _V = std::vector<int>(data.begin() + lumaSize + chromaSize, data.end());

  initBlockStructure();
}

void Frame::initBlockStructure(){
  _blockRows = _rows / _blockSize;
  _blockCols = _cols / _blockSize;

  _blockRowResidual = _rows % _blockSize;
  _blockColResidual = _cols % _blockSize;

  _blockRows += _blockRowResidual ? 1 : 0;
  _blockCols += _blockColResidual ? 1 : 0;
}

size_t Frame::getLumaIndex(size_t row, size_t col) const {
  if(_rows <= row){
    throw std::out_of_range(
      std::string("FRAME::LUMA::INDEX: row ") + std::to_string(row) +
      " is out of bound for frame with height " + std::to_string(_rows)
    );
  }
  if(_cols <= col){
    throw std::out_of_range(
      std::string("FRAME::LUMA::INDEX: col ") + std::to_string(col) +
      " is out of bound for frame with width " + std::to_string(_cols)
    );
  }
  return row * _cols + col;
}

size_t Frame::getChromaIndex(size_t row, size_t col) const {
  if(_rows <= row){
    throw std::out_of_range(
      std::string("FRAME::CHROMA::INDEX: row ") + std::to_string(row) +
      " is out of bound for frame with height " + std::to_string(_rows)
    );
  }
  if(_cols <= col){
    throw std::out_of_range(
      std::string("FRAME::CHROMA::INDEX: col ") + std::to_string(col) +
      " is out of bound for frame with width " + std::to_string(_cols)
    );
  }
  return (row / 2) * (_cols / 2) + (col / 2);
}


const std::vector<int> Frame::getY() const{
  return _Y;
}
const std::vector<int> Frame::getU() const{
  return _U;
}
const std::vector<int> Frame::getV() const{
  return _V;
}

int Frame::getYAt(size_t row, size_t col) const{
  return _Y[getLumaIndex(row, col)];
}
int Frame::getUAt(size_t row, size_t col) const{
  return _U[getChromaIndex(row, col)];
}
int Frame::getVAt(size_t row, size_t col) const{
  return _V[getChromaIndex(row, col)];
}

void Frame::setYAt(size_t row, size_t col, int value){
  _Y[getLumaIndex(row, col)] = value;
}
void Frame::setUAt(size_t row, size_t col, int value){
  _U[getChromaIndex(row, col)] = value;
}
void Frame::setVAt(size_t row, size_t col, int value){
  _V[getChromaIndex(row, col)] = value;
}

size_t Frame::getPixRows() const{
  return _rows;
}
size_t Frame::getPixCols() const{
  return _cols;
}

Block Frame::getBlock(size_t blockRow, size_t blockCol){
  if(_blockRows <= blockRow){
    throw std::out_of_range(
      std::string("FRAME::GETBLOCK::INDEX: block row ") + std::to_string(blockRow) +
      " is out of bound for frame with " + std::to_string(_blockRows) + " block rows."
    );
  }
  if(_blockCols <= blockCol){
    throw std::out_of_range(
      std::string("FRAME::GETBLOCK::INDEX: col ") + std::to_string(blockCol) +
      " is out of bound for frame with " + std::to_string(_blockCols) + " block columns."
    );
  }
  size_t width = blockCol == _blockCols - 1 ? (_blockColResidual ? _blockColResidual : _blockSize) : _blockSize;
  size_t height = blockRow == _blockRows - 1 ? (_blockRowResidual ? _blockRowResidual : _blockSize) : _blockSize;

  return Block::createBlock(this, blockRow * _blockSize, blockCol * _blockSize, width, height);
}


const Block Frame::getBlock(size_t blockRow, size_t blockCol) const{
  if(_blockRows <= blockRow){
    throw std::out_of_range(
      std::string("FRAME::GETBLOCK::INDEX: block row ") + std::to_string(blockRow) +
      " is out of bound for frame with " + std::to_string(_blockRows) + " block rows."
    );
  }
  if(_blockCols <= blockCol){
    throw std::out_of_range(
      std::string("FRAME::GETBLOCK::INDEX: col ") + std::to_string(blockCol) +
      " is out of bound for frame with " + std::to_string(_blockCols) + " block columns."
    );
  }
  size_t width = blockCol == _blockCols - 1 ? (_blockColResidual ? _blockColResidual : _blockSize) : _blockSize;
  size_t height = blockRow == _blockRows - 1 ? (_blockRowResidual ? _blockRowResidual : _blockSize) : _blockSize;

  return Block::createConstBlock(this, blockRow * _blockSize, blockCol * _blockSize, width, height);
}

size_t Frame::getBlockRows() const{
  return _blockRows;
}
size_t Frame::getBlockCols() const{
  return _blockCols;
}
size_t Frame::getBlockSize() const{
  return _blockSize;
}

void Frame::write(std::ostream& out) const{
  auto writeRaw = [&out](const std::vector<int>& data) {
    for (int val : data) {
      uint8_t byte = static_cast<uint8_t>(val);
      out.write(reinterpret_cast<const char*>(&byte), sizeof(uint8_t));
    }
  };

  writeRaw(_Y);
  writeRaw(_U);
  writeRaw(_V);
}

void Frame::write(const std::string& path) const{
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + path);
  }

  write(file);
}


