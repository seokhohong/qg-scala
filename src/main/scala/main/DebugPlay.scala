package main

import core.Bitboard
import learner.{Mind, SearchNode}

object DebugPlay {
  def main(args: Array[String]): Unit = {
    val board = new Bitboard(size=9, win_chain_length=5)
    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(board.make_move)

    val mind: Mind = new Mind("/models/v4_14_test", Map[String, Any](
      "min_child_p" -> -9.0,
      "p_batch_size" -> (1 << 10),
      "fraction_q" -> 1.0,
      "q_draw_weight" -> 0.0,
      "max_thinktime" -> 15.0,
      "num_pv_expand" -> 25
    ))
    val root_search = mind.make_move_debug(board)
    val best_move = root_search.best_move() getOrElse -1
    assert (best_move == 36)
    assert (root_search.get_q() == SearchNode.P1_WIN_Q)
  }
}