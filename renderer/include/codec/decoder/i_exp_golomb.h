/**
* Created by Brent Matthys on 10/04/2025
*/


#include <cstddef>
#include <cstdint>

class IExpGolomb {
private:
  uint32_t _data = 1;
  bool _parsing = false;
  size_t _leadingZeroes = 0;
  size_t _bitsParsed = 0;

  void clear();

  bool decodeLeading(bool bit, int& out);
  bool decodeParsing(bool bit, int& out);

  int dataToInt();

public:
  /**
  * This function should be called
  * consecutively bit by bit.
  *
  * If a symbol is detected it will be decoded
  * to out and this function will return true.
  * It will return false if no symbol is detected.
  */
  bool decode(bool bit, int& out);
};
