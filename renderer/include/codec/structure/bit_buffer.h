/**
* Created by Brent Matthys on 09/04/2025
*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class BitBuffer {
private:
  /* Buffer storing all completed bytes */
  std::vector<uint8_t> _data;

  /* Incomplete byte, storing bits */
  uint8_t _currentByte;

  /* The amount of bits stored in the current byte*/
  uint8_t _byteBitCount;
public:

  explicit BitBuffer();

  /**
  * Store a single bit.
  */
  void pushBit(bool bit);

  /**
  * Push back all bits from the lower `count` bits of `bits`
  *
  * Params:
  * `bits` uint32_t - The bits to store.
  * `count` uint8_t - The lower amount of bits from `bits` to store.
  */
  void pushBits(uint64_t bits, uint8_t count);

  /**
  * Push back all the bits of another bit buffer.
  */
  void pushBitBuffer(const BitBuffer& other);

  /**
  * Get the amount of bits stored in this bit buffer.
  */
  size_t getBitCount() const;

  /**
  * Get all the bits stored as a vector of bytes.
  * The byte is zero padded if getBitCount % 8 != 0.
  */
  std::vector<uint8_t> getBytes() const;

  bool getBit(size_t index) const;

  /**
  * Make this BitBuffer empty.
  */
  void clear();
};

