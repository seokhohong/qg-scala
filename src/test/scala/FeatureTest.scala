import org.scalatest.FunSuite
import core._
import learner._
import org.platanios.tensorflow.api.{Shape, Tensor}


class FeatureTest extends FunSuite {
  test("board features") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)
    fboard.make_move(40)
    val q_features = fboard.get_features()
    assert (q_features(40, 0) == 1)
    assert (q_features(40, 2) == 1)
    assert (q_features(25, 3) == 1)
  }
}
