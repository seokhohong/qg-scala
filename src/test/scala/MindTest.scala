import core.{MoveSeq, TranspositionTable}
import org.scalatest.FunSuite
import core.Bitboard
import learner.{Mind, SearchNode}

class MindTest extends FunSuite {
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
  test(testName = "simple win") {
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

    val root_search = mind.make_move_debug(board)
    val best_move = root_search.best_move() getOrElse -1
    // this is pretty temporary
    assert (root_search.get_q(SearchNode.P2_WIN_INDEX) < 0.5)
    assert (root_search.get_q(SearchNode.P2_WIN_INDEX) > 0.45)
    assert (root_search.pv_depth() == 3)
    assert (best_move == 36)
  }
  /*
      def test_thoughtboard_root_win(self):
        trivial_board = Board(size=9, win_chain_length=5)
        trivial_board.set_to_one_move_from_win()
        tb = ThoughtBoard(trivial_board, FeatureBoard_v1_1)

        trivial_board.move(36)
        print(trivial_board.pprint())
        move_list = MoveList((), []).append(36)
        self.assertTrue(trivial_board.game_won())
        self.assertTrue(tb.game_over_after(move_list))
   */
}
