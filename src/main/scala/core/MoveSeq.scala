package core

object MoveSeq {
  def empty(): MoveSeq = {
    MoveSeq()
  }
}
class MoveSeq(val moves: List[Int], val position_hash: List[Int]) {
  def append(new_move: Int): MoveSeq = {
    val sign = moves.size % 2 match {
      case 0 => 1
      case 1 => -1
    }
    MoveSeq(
      new_move :: moves,
      (sign * new_move :: position_hash).sorted
    )
  }
}