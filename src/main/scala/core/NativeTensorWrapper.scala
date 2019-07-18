package core

import java.nio.ByteOrder

import org.platanios.tensorflow.api.Tensor
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}
import java.nio.{ByteBuffer, FloatBuffer}

import breeze.linalg.{DenseMatrix, DenseVector}
import core.SimpleModel.{rm, ru}

import scala.reflect.ClassTag
import scala.reflect.runtime.universe._

//Native Buffer is a wrapper around the Tensorflow-native structures
object NativeTensorWrapper {
  // reflection access to fromHostNativeHandle private method
  private val ru = scala.reflect.runtime.universe

  // get runtime mirror
  private val rm = ru.runtimeMirror(getClass.getClassLoader)

  // get companion method symbol
  private val fromHostNativeSymbol = ru.typeOf[Tensor.type].decl(ru.TermName("fromHostNativeHandle")).asMethod

  // get method mirror
  private val fromHostNativeMirror: ru.MethodMirror = rm.reflect(Tensor).reflectMethod(fromHostNativeSymbol)


  //private val resolveSymbol = ru.typeOf[Tensor[Float]].decl(ru.TermName("resolve")).asMethod

  //private val resolveMethod = rm.reflect(Tensor).reflectMethod(resolveMethodSymbol)

  def extractMethod(tensor: Tensor[Float], methodName: String): MethodMirror = {
    val tensorClass = tensor.getClass
    val mirror = runtimeMirror(tensorClass.getClassLoader)
    val tensorType = mirror.classSymbol(tensorClass).toType
    val tensorClassTag = ClassTag[Tensor[Float]](mirror.runtimeClass(tensorType))
    val methodSymbol = tensorType
      .member(TermName(methodName)).asMethod
    mirror.reflect(tensor)(tensorClassTag)
      .reflectMethod(methodSymbol)
  }
  /*
  //proper reflection is pretty painful here, have to dig through two layers of classes
  def extractNativeHandleWrapper(tensor: Tensor[Float]): Long = {
    val tm = ru.runtimeMirror(tensor.getClass.getClassLoader)
    val wrapperSymb = ru.typeOf[Tensor[Float]].decl(ru.TermName("nativeHandleWrapper")).asTerm
    val wrapperMirror = tm.reflect(tensor).reflectField(wrapperSymb)
    shippingFieldMirror.get.asInstanceOf[NativeHandleWrapper]

    val m = ru.runtimeMirror(tensor.getClass.getClassLoader)
    val shippingTermSymb = ru.typeOf[Tensor[Float]].decl(ru.TermName("nativeHandleWrapper")).asTerm
    val im = m.reflect(tensor)
    val shippingFieldMirror = im.reflectField(shippingTermSymb)

  }
   */

  private val tensor = Tensor[Float](1.0.toFloat)
  private val resolveMethod = extractMethod(tensor, "resolve")



  final val BytesPerFloat = 4

  def allocate_floats(num_floats: Int): NativeTensorWrapper = {
    val allocated_buffer_handle = NativeTensor.allocate(1, Array[Long](num_floats), num_floats * NativeTensorWrapper.BytesPerFloat)
    new NativeTensorWrapper(allocated_buffer_handle)
  }
  /*
  def from_tensor(tensor: Tensor[Float]): NativeTensorWrapper = {

    //val resolvedHandle = resolveMethod(tensor).asInstanceOf[Long]
    new NativeTensorWrapper(extractNativeHandleWrapper(tensor))
  }
   */
}
class NativeTensorWrapper private(native_handle: Long) {
  val buffer = NativeTensor.buffer(native_handle).order(ByteOrder.nativeOrder)

  def putFloat(value: Float): ByteBuffer = buffer.putFloat(value)
  def putFloat(index: Int, value: Float): ByteBuffer = buffer.putFloat(index, value)
  def put(buff: NativeTensorWrapper): ByteBuffer = buffer.put(buff.buffer)
  // Returns the number of floats this buffer can hold
  def floatsCapacity: Int = buffer.capacity() / NativeTensorWrapper.BytesPerFloat

  def toArray: Array[Float] = {
    val result_buffer = buffer.asFloatBuffer()
    val result_array = Array.ofDim[Float](result_buffer.capacity())
    result_buffer.get(result_array)
    result_array
  }

  def toMatrix(num_obs: Int, num_output: Int): DenseMatrix[Float] = {
    DenseVector.create(toArray, 0, 0, num_obs * num_output).asDenseMatrix.reshape(num_obs, num_output)
  }

  def toFloatTensor: Tensor[Float] = NativeTensorWrapper.fromHostNativeMirror(native_handle).asInstanceOf[Tensor[Float]]
}
