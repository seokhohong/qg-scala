import breeze.linalg.{DenseVector, sum}
import core.Bitboard
import org.scalatest.FunSuite
import core._
import learner.FeatureBoard
import org.platanios.tensorflow.api.Tensor


class FeatureBoardTest extends FunSuite {
  // reflection access to fromHostNativeHandle private method
  private val ru = scala.reflect.runtime.universe

  // get runtime mirror
  private val rm = ru.runtimeMirror(getClass.getClassLoader)

  // get companion method symbol
  private val fromHostNativeSymbol = ru.typeOf[Tensor.type].decl(ru.TermName("fromHostNativeHandle")).asMethod

  // get method mirror
  private val fromHostNativeMirror: ru.MethodMirror = rm.reflect(Tensor).reflectMethod(fromHostNativeSymbol)

  test("board features") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)
    assert (fboard.get_features_as_nice_array()(4)(4)(3) == -1)
    fboard.make_move(40)
    val q_features = fboard.get_features_as_nice_array()
    assert (q_features(4)(4)(0) == 1)
    assert (q_features(4)(4)(2) == 1)
    assert (q_features(2)(7)(3) == 1)
    fboard.unmove()
    assert (fboard.is_reset())
  }
  test(testName = "player turn") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)
    assert (fboard.get_features_as_nice_array()(2)(7)(3) == -1)
    assert (fboard.is_reset())
    List[Int](0, 1, 9, 10, 18, 19, 27, 28).foreach(fboard.make_move)
    assert (fboard.get_features_as_nice_array()(2)(7)(3) == -1)
    fboard.unmove()
    assert (fboard.player_to_move() == Player.SECOND)
    assert (fboard.get_features_as_nice_array()(2)(7)(3) == 1)
    fboard.make_move(36)
    assert (fboard.get_features_as_nice_array()(2)(7)(3) == -1)
  }

  test(testName = "blank featureboard") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)

    val matrix = fboard.get_features_as_vector()
    for (i <- 0 until 81 * 4) {
      assert (matrix(i) <= 1)
      assert (matrix(i) >= -1)
    }
  }

  test(testName = "test move featureboard buffer") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)
    val buffer = NativeTensorWrapper.allocate_floats(81 * 4)

    fboard.make_move(10)

    fboard.write_features_inplace(buffer)

    val matrix = new DenseVector[Float](buffer.toFloatTensor.entriesIterator.toArray)
    assert (matrix(40) == 1)
    assert (matrix(42) == 1)
  }

  test(testName = "test multiwrite") {
    val board = new Bitboard()
    val fboard = new FeatureBoard(board)
    val buffer = NativeTensorWrapper.allocate_floats(81 * 4 * 2)

    fboard.make_move(10)

    fboard.write_features_inplace(buffer)
    fboard.write_features_inplace(buffer)

    val matrix = new DenseVector[Float](buffer.toFloatTensor.entriesIterator.toArray)
    assert (matrix(40) == 1)
    assert (matrix(42) == 1)

    assert (matrix(0 until 324) == matrix(324 until 324 * 2))
  }

}
