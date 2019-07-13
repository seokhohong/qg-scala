import core._
import org.scalatest.FunSuite
import breeze.linalg.DenseVector
import learner.SearchNode

class SearchNodeTest extends FunSuite {
  test( testName = "simple") {
    val node_params: Map[String, Any] = Map[String, Any]("q_draw_vector" -> DenseVector[Double](1, 0, -1))
    val base: SearchNode = SearchNode.make_root(is_maximizing = true, node_params)
    base.assign_leaf_q(DenseVector[Double](0.5, 0.3, 0.2))

    val long_win = base.create_child(16)
    assert (base.get_children().contains(long_win))
    assert (base.get_child(16) == long_win)
    assert (long_win.has_parents())

    assert (long_win.get_move_list() == MoveSeq.empty().append(16))
  }
  test("shortest win test") {
    val node_params: Map[String, Any] = Map[String, Any]("q_draw_vector" -> DenseVector[Double](1, 0, -1))
    val base: SearchNode = SearchNode.make_root(is_maximizing = true, node_params)
    base.assign_leaf_q(DenseVector[Double](0.5, 0.3, 0.2))
    assert (base.move_goodness() > 0)

    val long_win = base.create_child(16)
    long_win.assign_hard_q(GameState.WIN_PLAYER_1)
    long_win.assign_p(-1)
    assert (long_win.log_total_p == -1)
    base.update_pv()

    assert (base.pv() == long_win)
    assert (long_win.move_goodness() <= long_win.get_q(SearchNode.P1_WIN_INDEX))

    val long_win_2 = long_win.create_child(17)
    long_win_2.assign_hard_q(GameState.WIN_PLAYER_1)
    long_win_2.assign_p(-0.2)
    long_win.update_pv()
    assert (base.pv() == long_win_2)

    val long_win_3 = long_win_2.create_child(18)
    long_win_3.assign_hard_q(GameState.WIN_PLAYER_1)
    long_win_3.assign_p(-0.5)
    long_win_2.update_pv()
    assert (base.pv() == long_win_3)

    val short_loss = base.create_child(19)
    short_loss.assign_hard_q(GameState.WIN_PLAYER_2)
    short_loss.assign_p(-0.8)
    base.update_pv()
    assert (base.pv() == long_win_3)
    assert (base.pv().get_move_list().moves == base.pv().calculate_pv_order())

    val short_win = base.create_child(20)
    short_win.assign_hard_q(GameState.WIN_PLAYER_1)
    base.update_pv()
    assert (base.pv() == short_win)
    assert (base.pv().get_move_list().moves == base.pv().calculate_pv_order())
  }
}
