/**
* Created by Brent Matthys on 06/04/2025
*/
#pragma once

#include <cstddef>
#include <map>
#include <ostream>
#include <string>
#include <vector>

struct pixel {
  int y;
  int u;
  int v;
};

enum class Channel{
  Y,U,V
};

// Forward declaration
// Block is defined at the bottom of this file
// This is in the same file to prevent circular imports

/**
 * Class representing a macroblock of a frame.
 * This class doesn't store any data. Rather
 * it has a reference to the corresponding frame.
 * This class will update the data in the frame,
 * but trough the interface of a macroblock.
 */
class Block;

class Frame {
private:
  const size_t _rows;
  const size_t _cols;

  size_t _blockRows;
  size_t _blockCols;

  size_t _blockRowResidual;
  size_t _blockColResidual;

  const size_t _blockSize; // size of blocks
  
  std::vector<int> _Y;
  std::vector<int> _U;
  std::vector<int> _V;

  void initBlockStructure();

  size_t getLumaIndex(size_t row, size_t col) const;
  size_t getChromaIndex(size_t row, size_t col) const;

public:
  Frame(size_t width, size_t height, size_t blocksize);
  Frame(const std::string& path, size_t frame, size_t width, size_t height, size_t blocksize);

  const std::vector<int> getY() const;
  const std::vector<int> getU() const;
  const std::vector<int> getV() const;

  /**
   * Get the Y value of the pixel at the requested coordinate.
   */
  int getYAt(size_t row, size_t col) const;

  /**
   * Get the U value of the pixel at the requested coordinate.
   * This function abstracts chroma subsampling away.
   * You can just call the actual pixel coordinate.
   */
  int getUAt(size_t row, size_t col) const;

  /**
   * Get the V value of the pixel at the requested coordinate.
   * This function abstracts chroma subsampling away.
   * You can just call the actual pixel coordinate.
   */
  int getVAt(size_t row, size_t col) const;

  /**
   * Set the Y value of the pixel at the requested coordinate.
   */
  void setYAt(size_t row, size_t col, int value);

  /**
   * Set the U value of the pixel at the requested coordinate.
   * This function abstracts chroma subsampling away.
   * You can just call the actual pixel coordinate.
   */
  void setUAt(size_t row, size_t col, int value);

  /**
   * Set the V value of the pixel at the requested coordinate.
   * This function abstracts chroma subsampling away.
   * You can just call the actual pixel coordinate.
   */
  void setVAt(size_t row, size_t col, int value);

  size_t getPixRows() const;
  size_t getPixCols() const;

  /**
   * Get the macroblock at block coordinate.
   * This block is an interface to update that part
   * of this frame.
   */
  Block getBlock(size_t blockRow, size_t blockCol);
  const Block getBlock(size_t blockRow, size_t blockCol) const;

  size_t getBlockRows() const;
  size_t getBlockCols() const;

  size_t getBlockSize() const;

  void write(std::ostream& out) const;
  void write(const std::string& path) const;

  std::map<int, int> getFrequencyTable(){
    std::map<int, int> table;
    for(int coef : _Y){
      ++table[coef];
    }
    return table;
  }

  // --- Pixel Proxy for [] overloading ---
  // Alongside the frame [] overloading we can do:
  //
  // frame[{row, col}] = pixel{y,u,v}; // set
  // pixel p = frame[{row, col}]; // get
  class PixelRef {
  private:
    Frame& _frame;
    size_t _row, _col;

  public:
    PixelRef(Frame& frame, size_t row, size_t col)
      : _frame(frame), _row(row), _col(col) {}

    operator pixel() const {
      return pixel{
        _frame.getYAt(_row, _col),
        _frame.getUAt(_row, _col),
        _frame.getVAt(_row, _col)
      };
    }

    PixelRef& operator=(const pixel& p) {
      _frame.setYAt(_row, _col, p.y);
      _frame.setUAt(_row, _col, p.u);
      _frame.setVAt(_row, _col, p.v);
      return *this;
    }
  };

  // set pixel
  PixelRef operator[](std::pair<size_t, size_t> pos) {
    return PixelRef(*this, pos.first, pos.second);
  }

  // get pixel
  pixel operator[](std::pair<size_t, size_t> pos) const {
    return pixel{
      getYAt(pos.first, pos.second),
      getUAt(pos.first, pos.second),
      getVAt(pos.first, pos.second)
    };
  }
};

/**
 * Class representing a macroblock of a frame.
 * This class doesn't store any data. Rather
 * it has a reference to the corresponding frame.
 * This class will update the data in the frame,
 * but trough the interface of a macroblock.
 */
class Block{
private:
  size_t _frameRow; // row idx in the frame of the top left pixel in this block
  size_t _frameCol; // col idx in the frame of the top left pixel in this block

  size_t _width;
  size_t _height;

  // The frame that this block belongs to. Only valid if block is not const
  Frame* _frame = nullptr; 
  const Frame* _constFrame; // The frame that this block belongs to (readonly)
  
  void assertLumaIndices(size_t row, size_t col) const;
  void assertChromaIndices(size_t row, size_t col) const;

  Block(Frame* frame, size_t row, size_t col, size_t width, size_t height);
  Block(const Frame* frame, size_t row, size_t col, size_t width, size_t height);
public:

  /**
   * Construct a block.
   */
  static Block createBlock(Frame* frame, size_t row, size_t col, size_t width, size_t height);
  /**
   * Construct a const block.
   */
  static const Block createConstBlock(
    const Frame* frame, size_t row, size_t col, size_t width, size_t height
  );

  /**
  * Get the Y value of the requested pixel in this 16x16 block.
  */
  int getYAt(size_t row, size_t col) const;

  /**
  * Get the U value of the requested pixel in this 16x16 block.
  * Due to chroma subsampling, you should index 8x8.
  */
  int getUAt(size_t row, size_t col) const;

  /**
  * Get the V value of the requested pixel in this 16x16 block.
  * Due to chroma subsampling, you should index 8x8.
  */
  int getVAt(size_t row, size_t col) const;

  /**
   * Get one channel value of the requested pixel.
   * This will call getYAt, getUAt or getVAt depending
   * on the requested channel.
   */
  int getAt(size_t row, size_t col, Channel channel) const;


  /**
  * Set the Y value of the requested pixel in this 16x16 block
  */
  void setYAt(size_t row, size_t col, int value);

  /**
  * Set the U value of the requested pixel in this 16x16 block.
  * Due to chroma subsampling, you should index 8x8.
  */
  void setUAt(size_t row, size_t col, int value);

  /**
  * Set the U value of the requested pixel in this 16x16 block.
  * Due to chroma subsampling, you should index 8x8.
  */
  void setVAt(size_t row, size_t col, int value);

  /**
   * Set one channel value of the requested pixel.
   * This will call setYAt, setUAt or setVAt depending
   * on the requested channel.
   */
  void setAt(size_t row, size_t col, int value, Channel channel);

  size_t getLumaWidth() const;
  size_t getLumaHeight() const;

  size_t getChromaWidth() const;
  size_t getChromaHeight() const;

  size_t getWidth(Channel channel) const;
  size_t getHeight(Channel channel) const;
};
