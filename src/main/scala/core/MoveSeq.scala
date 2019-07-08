package core

object MoveSeq {
  def empty(): MoveSeq = {
    new MoveSeq(List[Int](), List[Int]())
  }
}
class MoveSeq(val moves: List[Int], val position_hash: List[Int]) {
  def append(new_move: Int): MoveSeq = {
    new MoveSeq(
      moves :+ new_move,
      hash_with_append(new_move)
    )
  }
  def hash_with_append(move: Int): List[Int] = {
    val sign = moves.size % 2 match {
      case 0 => 1
      case 1 => -1
    }
    (sign * move :: position_hash).sorted
  }

  def canEqual(a: Any) = a.isInstanceOf[MoveSeq]

  // Step 3 - proper signature for `equals`
  // Steps 4 thru 7 - implement a `match` expression
  override def equals(that: Any): Boolean =
    that match {
      case that: MoveSeq => {
        that.canEqual(this) &&
          this.moves == that.moves
      }
      case _ => false
    }

  // Step 8 - implement a corresponding hashCode c=method
  override def hashCode: Int = {
    position_hash.hashCode()
  }
}