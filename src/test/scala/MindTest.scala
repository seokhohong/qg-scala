import core.{MoveSeq, TranspositionTable}
import org.scalatest.FunSuite
import core.Bitboard
import learner.{Mind, SearchNode}

class MindTest extends FunSuite {
  test(testName = "simple win") {
    val board = new Bitboard(size=9, win_chain_length=5)
    List[Int](54, 63, 28, 22, 3, 31, 24, 40, 32, 23, 41, 25, 50, 13).foreach(board.make_move)

    val mind: Mind = new Mind("/models/v4_15", Map[String, Any](
      "min_child_p" -> -7.0,
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
    val root_search = searcher.root_node
    val best_move = root_search.best_move() getOrElse -1
    // this is pretty temporary
    assert (root_search.pv_depth() == 2)
    assert (root_search.get_q() == SearchNode.P2_WIN_Q)
  }
  test(testName = "simple win one more step") {
    val board = new Bitboard(size=9, win_chain_length=5)
    List[Int](54, 63, 28, 22, 3, 31, 24, 40, 32, 23, 41, 25, 50).foreach(board.make_move)

    val mind: Mind = new Mind("/models/v4_15", Map[String, Any](
      "min_child_p" -> -7.0,
      "p_batch_size" -> (1 << 10),
      "fraction_q" -> 1.0,
      "q_draw_weight" -> 0.0,
      "max_thinktime" -> 15.0,
      "num_pv_expand" -> 25
    ))

    val searcher = mind.make_search(board)
    searcher.run_iteration()
    searcher.run_iteration()
    searcher.run_iteration()
    val root_search = searcher.root_node
    val best_move = root_search.best_move() getOrElse -1
    // this is pretty temporary
    assert (root_search.get_q() == SearchNode.P2_WIN_Q)
    assert (root_search.pv_depth() == 3)
  }
  test(testName = "root_win") {
    val board = new Bitboard(size=9, win_chain_length=5)
    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(board.make_move)

    val mind: Mind = new Mind("/models/v4_15", Map[String, Any](
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
  def just_no_crash_from_position(mind: Mind, moves: List[Int], num_iterations: Int): Unit ={
    val board = new Bitboard(size=9, win_chain_length=5)
    moves.foreach(board.make_move)
    val searcher = mind.make_search(board)
    searcher.run_iterations(num_iterations)
  }
  test(testName = "no crash") {

    val mind: Mind = new Mind("/models/v4_15", Map[String, Any](
      "min_child_p" -> -9.0,
      "p_batch_size" -> (1 << 10),
      "fraction_q" -> 1.0,
      "q_draw_weight" -> 0.0,
      "max_thinktime" -> 15.0,
      "num_pv_expand" -> 25
    ))
    var moves = List[Int](37, 45, 74, 50, 47, 29, 31, 39, 49, 58, 48, 41, 40, 32, 22, 13, 23, 59, 68, 21, 60, 36, 5, 42, 34, 20,
      67, 57, 56, 65, 52, 44, 38, 12, 30, 14, 11, 15, 16, 43, 46, 54, 66, 18, 27, 72, 63, 28, 4, 69, 76, 55,
      75, 77, 25, 7, 24, 26, 53, 61, 78, 73, 19, 8, 64, 35, 62, 79, 51, 6, 17, 3)

    just_no_crash_from_position(mind, moves, 2)

    moves = List[Int](70, 69, 10, 42, 43, 25, 50, 40, 49, 48, 32, 41, 39, 58, 38, 31, 51, 22, 13, 52, 21, 29, 67, 59, 56, 37,
      61, 30, 28, 47, 12, 14, 11, 9, 33, 3, 23, 53, 27, 24, 6, 46, 57, 19, 55, 2, 44, 34, 4, 74, 7, 36, 5, 8,
      54, 62, 16, 65, 66, 78, 75, 68, 1, 15, 76, 77, 26, 35, 45, 63, 18, 60, 79, 20, 0, 64, 80, 72, 73, 17)
    just_no_crash_from_position(mind, moves, 2)

    moves = List[Int](23, 66, 44, 80, 35, 12, 33, 30, 31, 39, 40, 48, 75)
    just_no_crash_from_position(mind, moves, 2)

    moves = List[Int](41, 8, 2, 29, 73, 31, 49, 40, 33, 50, 30, 39, 57, 22)
    just_no_crash_from_position(mind, moves, 2)
  }
  test(testName = "move order search") {
    val mind: Mind = new Mind("/models/v4_15", Map[String, Any](
      "min_child_p" -> -9.0,
      "p_batch_size" -> (1 << 10),
      "fraction_q" -> 1.0,
      "q_draw_weight" -> 0.0,
      "max_thinktime" -> 15.0,
      "num_pv_expand" -> 25
    ))

    val board = new Bitboard(size=9, win_chain_length=5)

    List[Int](64, 1, 66, 16, 67, 28, 29, 37, 31, 55, 57, 56, 47, 79, 70, 62, 53, 15, 25).foreach(board.make_move)
    val searcher = mind.make_search(board)
    searcher.run_iterations(3)

    assert (searcher.get_pv().calculate_pv_order() == List[Int](19, 46, 10))
  }
}
