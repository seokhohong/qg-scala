import core._
import learner._
import org.scalatest.FunSuite
import core.SimpleModel
import org.platanios.tensorflow.api.{Tensor, Shape}

class ModelTest extends FunSuite {
  test("load model") {
    val model = new SimpleModel("/models/v4_12_value.pb", "input_1:0", "dense_5/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    val tensor_input = Tensor[Float](fboard.get_q_features().toArray).reshape(Shape(1, 9, 9, 4))
    assert (model.predict(tensor_input).head(0) < 0.38)
  }
}
