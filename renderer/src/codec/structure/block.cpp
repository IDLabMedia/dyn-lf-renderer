/**
* Created by Brent Matthys on 06/04/2025
*/

#include "codec/structure/frame.h"
#include <cstddef>
#include <stdexcept>
#include <string>

Block::Block(Frame* frame, size_t row, size_t col, size_t width, size_t height):
  _frame(frame), _constFrame(frame), _frameRow(row), _frameCol(col), _width(width), _height(height){}


Block::Block(const Frame* frame, size_t row, size_t col, size_t width, size_t height):
  _constFrame(frame), _frameRow(row), _frameCol(col), _width(width), _height(height){}


Block Block::createBlock(Frame* frame, size_t row, size_t col, size_t width, size_t height){
  return Block(frame, row, col, width, height);
}
const Block Block::createConstBlock(const Frame* frame, size_t row, size_t col, size_t width, size_t height){
  return Block(frame, row, col, width, height);
}


void Block::assertLumaIndices(size_t row, size_t col) const{
  if(_height <= row){
    throw std::out_of_range(
      std::string("BLOCK::LUMA::INDEX: row ") + std::to_string(row) +
      " is out of bound for block with height " + std::to_string(_height)
    );
  }
  if(_width <= col){
    throw std::out_of_range(
      std::string("BLOCK::LUMA::INDEX: col ") + std::to_string(col) +
      " is out of bound for block with width " + std::to_string(_width)
    );
  }
}

void Block::assertChromaIndices(size_t row, size_t col) const{
  if(getChromaHeight() <= row){
    throw std::out_of_range(
      std::string("BLOCK::CHROMA::INDEX: row ") + std::to_string(row) +
      " is out of bound for block with height " + std::to_string(getChromaHeight())
    );
  }
  if(getChromaWidth() <= col){
    throw std::out_of_range(
      std::string("BLOCK::CHROMA::INDEX: col ") + std::to_string(col) +
      " is out of bound for block with width " + std::to_string(getChromaWidth())
    );
  }
}


int Block::getYAt(size_t row, size_t col) const{
  assertLumaIndices(row, col);
  return _constFrame->getYAt(_frameRow + row, _frameCol + col);
}
int Block::getUAt(size_t row, size_t col) const{
  assertChromaIndices(row, col);
  return _constFrame->getUAt(_frameRow + (row*2), _frameCol + (col*2));
}
int Block::getVAt(size_t row, size_t col) const{
  assertChromaIndices(row, col);
  return _constFrame->getVAt(_frameRow + (row*2), _frameCol + (col*2));
}
int Block::getAt(size_t row, size_t col, Channel channel) const{
  if(channel == Channel::Y){
    return getYAt(row, col);
  }
  if(channel == Channel::U){
    return getUAt(row, col);
  }
  return getVAt(row, col);
}

void Block::setYAt(size_t row, size_t col, int value){
  assertLumaIndices(row, col);
  _frame->setYAt(_frameRow + row, _frameCol + col, value);
}
void Block::setUAt(size_t row, size_t col, int value){
  assertChromaIndices(row, col);
  _frame->setUAt(_frameRow + (row*2), _frameCol + (col*2), value);
}
void Block::setVAt(size_t row, size_t col, int value){
  assertChromaIndices(row, col);
  _frame->setVAt(_frameRow + (row*2), _frameCol + (col*2), value);
}
void Block::setAt(size_t row, size_t col, int value, Channel channel){
  if(channel == Channel::Y){
    setYAt(row, col, value);
  }else if(channel == Channel::U){
    setUAt(row, col, value);
  }else{
    setVAt(row, col, value);
  }
}

size_t Block::getLumaWidth() const{
  return _width;
}
size_t Block::getLumaHeight() const{
  return _height;
}

size_t Block::getChromaWidth() const{
  return _width / 2;
}
size_t Block::getChromaHeight() const{
  return _height / 2;
}

size_t Block::getWidth(Channel channel) const{
  return channel == Channel::Y ? getLumaWidth() : getChromaWidth();
}

size_t Block::getHeight(Channel channel) const{
  return channel == Channel::Y ? getLumaHeight() : getChromaHeight();
}
