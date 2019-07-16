import core._
import learner._
import org.scalatest.FunSuite
import core.SimpleModel
import scala.collection.mutable
import org.platanios.tensorflow.api.{Tensor, Shape}
import org.platanios.tensorflow.api.tensors.ops.Basic
import breeze.linalg.DenseMatrix

class ModelTest extends FunSuite {
  /*
  test("load model") {
    val model = new SimpleModel("/models/v4_14_test_value.pb", "input_tensor:0", "softmax_output/Softmax:0")
    val model2 = new SimpleModel("/models/v4_14_test_policy.pb", "tensor_input:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    val tensor_input = fboard.get_q_features().reshape(Shape(1, 9, 9, 4))
    assert (model.predict(tensor_input)(0, 0) > 0.3)
  }

  test("test p model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)

    val tensor_input = fboard.get_p_features().reshape(Shape(1, 9, 9, 4))

    assert (tensor_input(0)(3)(1).entriesIterator.toArray sameElements Array[Int](0, 1, 1, -1))

    assert (model.predict(tensor_input)(0, 36) > 0.5)
  }

  test("stack p model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)

    val features = new mutable.ListBuffer[Tensor[Float]]()
    features += fboard.get_p_features()
    features += fboard.get_p_features()

    val tensor_input = Tensor[Float](Basic.stack(features)).reshape(Shape(2, 9, 9, 4))

    assert (model.predict(tensor_input).rows == 2)
    assert (model.predict(tensor_input).cols == 81)
  }

  test("validate output orientation model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard_1 = new FeatureBoard(board)
    val fboard_2 = new FeatureBoard(board)

    List[Int](54, 63, 28, 22, 3, 31, 24, 40, 32, 23, 41, 25, 50, 13, 4).foreach(fboard_1.make_move)
    List[Int](54, 63, 28, 22, 3, 31, 24, 40, 32, 23, 41, 25, 50, 13, 49).foreach(fboard_2.make_move)

    val features = new mutable.ListBuffer[Tensor[Float]]()
    features += fboard_1.get_p_features()
    features += fboard_2.get_p_features()

    val tensor_input = Tensor[Float](Basic.stack(features)).reshape(Shape(2, 9, 9, 4))

    val predictions = model.predict(tensor_input)

    assert(predictions(0, 49) > 0.5)
    assert(predictions(1, 4) > 0.5)
  }
*/
}
