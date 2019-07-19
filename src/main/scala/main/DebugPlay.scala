package main

import core.Bitboard
import learner.{Mind, SearchNode}

object DebugPlay {
  def main(args: Array[String]): Unit = {
    val board = new Bitboard(size=9, win_chain_length=5)
    List[Int](54, 63, 28, 22, 15, 34, 67, 44, 35).foreach(board.make_move)

    val mind: Mind = new Mind("/models/v4_15", Map[String, Any](
      "min_child_p" -> -6.0,
      "p_batch_size" -> (1 << 12),
      "fraction_q" -> 1.0,
      "q_draw_weight" -> 0.0,
      "max_thinktime" -> 120.0,
      "num_pv_expand" -> 200,
      "confidence_function" -> ((x: Int) => 0.05 * math.pow(1.1, x))
    ), validate=true)
    println(board)
    val searcher = mind.make_search(board)
    searcher.run_iterations(15)
    println(searcher.root_node.best_move())
    println(searcher.compute_confidence_pv())
    val goodness = searcher.root_node.compute_search_goodness()
    println(goodness.composite_score)
  }
}