import core._
import learner._
import org.scalatest.FunSuite
import core.SimpleModel
import scala.collection.mutable
import org.platanios.tensorflow.api.{Tensor, Shape}
import org.platanios.tensorflow.api.tensors.ops.Basic

class ModelTest extends FunSuite {
  test("load model") {
    val model = new SimpleModel("/models/v4_14_test_value.pb", "input_tensor:0", "softmax_output/Softmax:0")
    val model2 = new SimpleModel("/models/v4_14_test_policy.pb", "tensor_input:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    val tensor_input = Tensor[Float](fboard.get_q_features().t.toArray).reshape(Shape(1, 9, 9, 4))
    assert (model.predict(tensor_input)(0, 0) > 0.3)
  }

  test("test p model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)

    val tensor_input = Tensor[Float](fboard.get_p_features().t.toArray).reshape(Shape(1, 9, 9, 4))

    assert (tensor_input(0)(3)(1).entriesIterator.toArray sameElements Array[Int](0, 1, 1, -1))

    assert (model.predict(tensor_input)(0, 36) > 0.5)
  }

  test("stack p model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)

    val features = new mutable.ListBuffer[Tensor[Float]]()
    features += fboard.get_p_features().t.toArray
    features += fboard.get_p_features().t.toArray

    val tensor_input = Tensor[Float](Basic.stack(features)).reshape(Shape(2, 9, 9, 4))

    assert (model.predict(tensor_input).rows == 2)
    assert (model.predict(tensor_input).cols == 81)
  }
}
