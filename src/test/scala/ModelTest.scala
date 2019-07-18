import java.nio.ByteBuffer

import core._
import learner._
import org.scalatest.FunSuite
import core.SimpleModel

import scala.collection.mutable
import org.platanios.tensorflow.api.{Shape, Tensor}
import org.platanios.tensorflow.api.tensors.ops.Basic
import breeze.linalg.DenseMatrix

class ModelTest extends FunSuite {
  /*
  test("test unsafe") {
    val model = new SimpleModel("/models/v4_14_test_value.pb", "input_tensor:0", "softmax_output/Softmax:0")
    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    val buffer = NativeTensorWrapper.allocate_floats(324)
    fboard.write_p_features_inplace(buffer)

    //val prediction = model.predict_to_matrix_unsafe(buffer, 1, 81)
    val prediction = model.predict_to_matrix_unsafe(buffer)
    assert (prediction(0, 0) > 0.3)
    assert (prediction(0, 0) < 0.4)
  }
  */
  test("load model") {
    val model = new SimpleModel("/models/v4_14_test_value.pb", "input_tensor:0", "softmax_output/Softmax:0")
    val model2 = new SimpleModel("/models/v4_14_test_policy.pb", "tensor_input:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    val buffer = NativeTensorWrapper.allocate_floats(324)
    fboard.write_p_features_inplace(buffer)
    assert (model.predict_to_matrix(buffer)(0, 0) > 0.3)
    assert (model.predict_to_matrix(buffer)(0, 0) < 0.4)
  }

  test("test p model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)

    val buffer = NativeTensorWrapper.allocate_floats(324)
    fboard.write_p_features_inplace(buffer)

    assert (model.predict_to_array(buffer)(36) > 0.5)
  }

  test("stack p model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard = new FeatureBoard(board)

    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)

    val buffer = NativeTensorWrapper.allocate_floats(324 * 2)
    fboard.write_p_features_inplace(buffer)
    fboard.write_p_features_inplace(buffer)

    val pred_matrix = model.predict_to_matrix(buffer)
    assert (pred_matrix.rows == 2)
    assert (pred_matrix.cols == 81)
    assert (pred_matrix(0, ::) == pred_matrix(1, ::))
  }
  test("validate output orientation model") {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val board = new Bitboard(size=9, win_chain_length = 5)
    val fboard_1 = new FeatureBoard(board)
    val fboard_2 = new FeatureBoard(board)

    List[Int](54, 63, 28, 22, 3, 31, 24, 40, 32, 23, 41, 25, 50, 13, 4).foreach(fboard_1.make_move)
    List[Int](54, 63, 28, 22, 3, 31, 24, 40, 32, 23, 41, 25, 50, 13, 49).foreach(fboard_2.make_move)

    val buffer = NativeTensorWrapper.allocate_floats(324 * 2)
    fboard_1.write_p_features_inplace(buffer)
    fboard_2.write_p_features_inplace(buffer)

    val predictions = model.predict_to_matrix(buffer)

    assert(predictions(0, 49) > 0.5)
    assert(predictions(1, 4) > 0.5)
  }
}
