import org.scalatest.FunSuite
import core._
import learner._
import org.platanios.tensorflow.api.{Shape, Tensor}


class FeatureBoardTest extends FunSuite {
  test("board features") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)
    assert (fboard.get_features()(25, 3) == -1)
    fboard.make_move(40)
    val q_features = fboard.get_features()
    assert (q_features(40, 0) == 1)
    assert (q_features(40, 2) == 1)
    assert (q_features(25, 3) == 1)
    fboard.unmove()
    assert (fboard.is_reset())
  }
  test(testName = "player turn") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)
    assert (fboard.get_features()(25, 3) == -1)
    assert (fboard.is_reset())
    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)
    assert (fboard.get_features()(25, 3) == -1)
    fboard.unmove()
    assert (fboard.player_to_move() == Player.SECOND)
    assert (fboard.get_features()(25, 3) == 1)
    fboard.make_move(36)
    assert (fboard.get_features()(25, 3) == -1)
  }
}
