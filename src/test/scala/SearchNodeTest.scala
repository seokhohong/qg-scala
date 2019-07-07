import core._
import org.scalatest.FunSuite
import breeze.linalg.DenseVector
import learner.SearchNode

class SearchNodeTest extends FunSuite {
  test("shortest win test") {
    val node_params: Map[String, Any] = Map[String, Any]("q_draw_vector" -> DenseVector[Double](1, 0, -1))
    val base: SearchNode = SearchNode.make_root(is_maximizing = true, node_params)
    base.assign_leaf_q(DenseVector[Double](0.5, 0.3, 0.2))

    val long_win = base.create_child(16)
    long_win.assign_hard_q(GameState.WIN_PLAYER_1)
    long_win.assign_p(-1)

    assert (base.pv() == long_win)
    assert (long_win.move_goodness() <= long_win.get_q(SearchNode.P1_WIN_INDEX))

    val long_win_2 = long_win.create_child(17)
    long_win_2.assign_hard_q(GameState.WIN_PLAYER_1)
    long_win_2.assign_p(-0.2)
    assert (base.pv() == long_win_2)

    val long_win_3 = long_win_2.create_child(18)
    long_win_3.assign_hard_q(GameState.WIN_PLAYER_1)
    long_win_3.assign_p(-0.5)
    assert (base.pv() == long_win_3)

    val short_loss = base.create_child(19)
    short_loss.assign_hard_q(GameState.WIN_PLAYER_2)
    short_loss.assign_p(-0.8)
  }
  /*
      def test_shorter_win(self):
        node_params = {'q_draw_vector': np.array([1, 0, -1])}
        base = PExpNodeV4(parent=None, move=15, params=node_params)

        base.assign_leaf_q((0.5, 0.3, 0.2))

        long_win = PExpNodeV4(base, 16, params=node_params)
        long_win.assign_hard_q(GameState_v2.WIN_PLAYER_1)
        long_win.assign_p(-1)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win)
        # check move_goodness looks at length too
        self.assertLess(long_win._move_goodness, long_win.get_q()[PExpNodeV4.P1_WIN_INDEX])

        long_win_2 = PExpNodeV4(long_win, 17, params=node_params)
        long_win_2.assign_hard_q(GameState_v2.WIN_PLAYER_1)
        long_win_2.assign_p(-0.2)
        long_win.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win_2)

        long_win_3 = PExpNodeV4(long_win_2, 18, params=node_params)
        long_win_3.assign_hard_q(GameState_v2.WIN_PLAYER_1)
        long_win_3.assign_p(-0.5)
        long_win_2.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win_3)

        short_loss = PExpNodeV4(base, 19, params=node_params)
        short_loss.assign_hard_q(GameState_v2.WIN_PLAYER_2)
        short_loss.assign_p(-0.8)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), long_win_3)
        self.assertEqual(list(base.get_principal_variation().get_move_chain()),
                         base.get_principal_variation().calculate_pv_order())

        short_win = PExpNodeV4(base, 20, params=node_params)
        short_win.assign_hard_q(GameState_v2.WIN_PLAYER_1)
        base.recalculate_q()
        self.assertEqual(base.get_principal_variation(), short_win)

        self.assertEqual(base.best_child, short_win)
        self.assertEqual(list(base.get_principal_variation().get_move_chain()),
                         base.get_principal_variation().calculate_pv_order())
   */
}
