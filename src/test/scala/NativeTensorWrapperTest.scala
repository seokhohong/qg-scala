import java.nio.{ByteBuffer, ByteOrder}

import breeze.linalg.DenseVector
import core.NativeTensorWrapper.getClass
import breeze.linalg.sum
import learner.FeatureBoard
import org.platanios.tensorflow.api.Tensor
import org.scalatest.FunSuite
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import core._

class NativeTensorWrapperTest extends FunSuite {

  // reflection access to fromHostNativeHandle private method
  private val ru = scala.reflect.runtime.universe

  // get runtime mirror
  private val rm = ru.runtimeMirror(getClass.getClassLoader)

  // get companion method symbol
  private val fromHostNativeSymbol = ru.typeOf[Tensor.type].decl(ru.TermName("fromHostNativeHandle")).asMethod

  // get method mirror
  private val fromHostNativeMirror: ru.MethodMirror = rm.reflect(Tensor).reflectMethod(fromHostNativeSymbol)

  test("buffer test") {

    val num_obs = 1
    val size = 9
    val channels = 4
    val bytes_per_float = 4
    val hostHandle = NativeTensor.allocate(1, Array[Long](num_obs, size * size, channels),
      num_obs * size * size * channels * bytes_per_float)
    val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)

    //clear buffer
    for (i <- 0 until size * size * channels) {
      buffer.putFloat(0)
    }
    buffer.clear()

    buffer.putFloat(10)

    val tensor: Tensor[Float] = fromHostNativeMirror(hostHandle).asInstanceOf[Tensor[Float]]

    val matrix = new DenseVector[Float](tensor.entriesIterator.toArray)
    assert (matrix(0) == 10)
    assert (sum(matrix) == 10)
  }
  test("NativeBuffer") {
    val buffer = NativeTensorWrapper.allocate_floats(10)

    assert (sum(new DenseVector[Float](buffer.toFloatTensor.entriesIterator.toArray)) == 0)
    buffer.putFloat(10)
    buffer.putFloat(4, 5)
    buffer.putFloat(8, 7)
    buffer.putFloat(12, 8)

    val tensor: Tensor[Float] = buffer.toFloatTensor
    val matrix = new DenseVector[Float](tensor.entriesIterator.toArray)
    assert (matrix(0) == 10)
    assert (matrix(1) == 5)
    assert (matrix(2) == 7)
    assert (matrix(3) == 8)
  }
  /*
  test("from tensor to array") {
    val tensor = Tensor[Float](2.0.toFloat, 5.0.toFloat)
    val nativetensor = NativeTensorWrapper.from_tensor(tensor)
    assert (nativetensor.toArray.length == 2)
    assert (nativetensor.toArray.apply(1) == 5)
  }
  */
}
