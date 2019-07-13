package main

import core.Bitboard
import learner.{Mind, SearchNode}

object DebugPlay {
  def main(args: Array[String]): Unit = {
    val board = new Bitboard(size=9, win_chain_length=5)
    List[Int](54, 63, 28, 22, 3, 31, 24, 40, 32, 23, 41, 25, 50).foreach(board.make_move)

    val mind: Mind = new Mind("/models/v4_15", Map[String, Any](
      "min_child_p" -> -9.0,
      "p_batch_size" -> (1 << 10),
      "fraction_q" -> 1.0,
      "q_draw_weight" -> 0.0,
      "max_thinktime" -> 15.0,
      "num_pv_expand" -> 25
    ))
    println(board)
    val searcher = mind.make_search(board)
    searcher.run_iteration()
    searcher.run_iteration()
    println(searcher.root_node.best_move())

  }
}