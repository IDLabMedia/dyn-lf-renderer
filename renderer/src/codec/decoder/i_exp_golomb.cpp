/**
* Created by Brent Matthys on 10/04/2025
*/


#include "codec/decoder/i_exp_golomb.h"

void IExpGolomb::clear(){
  _data = 1; // the center one we will always have
  _parsing = false;
  _leadingZeroes = 0;
  _bitsParsed = 0;
}

bool IExpGolomb::decodeLeading(bool bit, int& out){
  if(!bit){ // zero bit, keep counting
    ++_leadingZeroes;
    return false;
  }
  // got a one bit: handle it
  if(_leadingZeroes == 0){ // no previous zeroes => output 0
    out = 0; // output 0 value
    clear(); // reset for next code
    return true; // signal value found
  }
  _parsing = true; // we can start parsing
  return false;
}

bool IExpGolomb::decodeParsing(bool bit, int& out){
  _data = (_data << 1) | bit; // insert the bit
  ++_bitsParsed; // keep track of that bit

  if(_bitsParsed != _leadingZeroes) return false; // keep parsing

  // done parsing
  _data -= 1; // subtract by one
  out = dataToInt(); // output data
  clear(); // reset for next code
  return true;
}

int IExpGolomb::dataToInt(){
  // if odd => last bit = 1 => make netagive
  return (_data & 1) ? (int)((_data + 1) / 2) : -(int)(_data / 2);
}

bool IExpGolomb::decode(bool bit, int& out){
  return _parsing ? decodeParsing(bit, out) : decodeLeading(bit, out);
}

