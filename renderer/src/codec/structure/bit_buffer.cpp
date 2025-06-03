/**
* Created by Brent Matthys on 09/04/2025
*/


#include "codec/structure/bit_buffer.h"
#include <cstddef>
#include <cstdint>
#include <vector>

BitBuffer::BitBuffer(): _currentByte(0), _byteBitCount(0){}

void BitBuffer::pushBit(bool bit){
  _currentByte <<= 1; // make room for 1 bit at lsb
  _currentByte |= (bit ? 1 : 0); // fill lsb with the value
  ++_byteBitCount; // keep track that we stored a bit

  if (_byteBitCount == 8) { // flush byte to data if byte is full
      _data.push_back(_currentByte);
      _currentByte = 0;
      _byteBitCount = 0;
  }
}

void BitBuffer::pushBits(uint64_t bits, uint8_t count){
  for (int i = count - 1; i >= 0; --i) {
    pushBit((bits >> i) & 1);
  }
}


void BitBuffer::pushBitBuffer(const BitBuffer& other){
  auto bytes = other.getBytes();
  size_t totalBits = other.getBitCount();

  for(size_t i = 0; i < bytes.size() - 1; ++i){
    pushBits(bytes[i], 8);
  }
  // Handle the last (possibly partial) byte correctly
  // values are stored from left -> right (padding on the right)
  // pushBits pushes the `count` RIGHTmost bits from left -> right
  // We want to push the LEFTmost bits from left -> right
  size_t remainingBits = totalBits % 8;
  if(remainingBits == 0){ // actually 8 bits left
    pushBits(bytes.back(), 8);
  }else{ // 1-7 bits left
    uint8_t lastByte = bytes.back();
    // Shift and push bits from LSB to MSB, just like in `pushBits()`
    for (int i = 7; i >= (8 - remainingBits); --i) {
      pushBit((lastByte >> i) & 1);  // Push each bit from MSB to LSB
    }
  }
}

size_t BitBuffer::getBitCount() const {
  return _data.size() * 8 + _byteBitCount;
}

std::vector<uint8_t> BitBuffer::getBytes() const {
  std::vector<uint8_t> out = _data;
  if(_byteBitCount > 0){
    uint8_t paddedByte = _currentByte << (8 - _byteBitCount);
    out.push_back(paddedByte);
  }
  return out;
}

bool BitBuffer::getBit(size_t index) const{
  size_t byteIndex = index / 8;
  size_t bitIndex = index % 8;

  auto data = getBytes();
  return (data[byteIndex] >> (7 - bitIndex)) & 1;
}

void BitBuffer::clear(){
  _data.clear();
  _currentByte = 0;
  _byteBitCount = 0;
}

