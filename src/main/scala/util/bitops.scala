package util

import collection.immutable

object bitops {
  def has_consecutive_bits(bitset: immutable.BitSet, num_consecutive: Int): Boolean = {
    var consecutive = 0
    for (i <- 0 until bitset.size) {
      if (bitset.contains(i)) {
        consecutive += 1
      }
      else {
        consecutive = 0
      }
      if (consecutive == num_consecutive) {
        return true
      }
    }
    false
  }
}
