#include <cstdint>
#include <iostream>
#include <iterator>
#include <ostream>
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "codec/encoder/exp_golomb.h"
#include "codec/decoder/i_exp_golomb.h"
#include "codec/structure/bit_buffer.h"


ExpGolomb encoder;

int expGolombEncodeDecode(int testVal){
  IExpGolomb decoder;

  BitBuffer encoded = encoder.expGolomb(testVal);

  int decoded = 0;
  for (size_t i = 0; i < encoded.getBitCount(); ++i) {
    bool bit = encoded.getBit(i);
    decoder.decode(bit, decoded); // only at the last bit should decoded be set
  }

  // Return decoded value for comparison
  return decoded;
}

uint64_t bitBuffertoUint64_t(BitBuffer buf){
  uint64_t val = 0;
  for (size_t i = 0; i < buf.getBitCount(); ++i) {
    bool bit = buf.getBit(i);
    val <<= 1;
    val |= bit;
  }
  return val;

}

/*
* only encode values that wont result in codes longer then 64 bits
*/
uint64_t getExpGolombEncode(int testVal){
  BitBuffer encoded = encoder.expGolomb(testVal);
  return bitBuffertoUint64_t(encoded);
}

TEST_SUITE("Exponential Golomb"){
  TEST_CASE("Simple encode"){
    CHECK(getExpGolombEncode(0) == 0b1);

    CHECK(getExpGolombEncode(1) == 0b010);
    CHECK(getExpGolombEncode(-1) == 0b011);

    CHECK(getExpGolombEncode(2) == 0b00100);
    CHECK(getExpGolombEncode(-2) == 0b00101);

    CHECK(getExpGolombEncode(3) == 0b00110);
    CHECK(getExpGolombEncode(-3) == 0b00111);

    CHECK(getExpGolombEncode(4) == 0b0001000);
    CHECK(getExpGolombEncode(-4) == 0b0001001);
  }

  TEST_CASE("BitBuffer block merge"){
    auto a = encoder.expGolomb(1); 
    auto b = encoder.expGolomb(-1); 
    a.pushBitBuffer(b);
    CHECK(bitBuffertoUint64_t(a) == 0b010011);
  }

  TEST_CASE("Single"){
    CHECK(expGolombEncodeDecode(0) == 0);
    CHECK(expGolombEncodeDecode(1) == 1);
    CHECK(expGolombEncodeDecode(-1) == -1);
    CHECK(expGolombEncodeDecode(2) == 2);
    CHECK(expGolombEncodeDecode(-2) == -2);
  }

  TEST_CASE("Uint8 values"){
    for (int i = 0; i < 256; ++i) {
      CHECK(expGolombEncodeDecode(i) == i);
    }
  }

  TEST_CASE("Range [-4096, 4095]"){
    for (int i = -4096; i < 4096; ++i) {
      CHECK(expGolombEncodeDecode(i) == i);
    }
  }

}
