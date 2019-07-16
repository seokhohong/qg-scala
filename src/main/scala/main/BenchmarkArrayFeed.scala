package main

import java.util.Calendar

import core.{Bitboard, SimpleModel}
import learner.FeatureBoard
import org.platanios.tensorflow.api.tensors.ops.Basic
import org.platanios.tensorflow.jni.{Tensor => NativeTensor, TensorFlow => NativeLibrary}
import org.platanios.tensorflow.api.{Shape, Tensor}
import org.platanios.tensorflow.api.core.types
import org.platanios.tensorflow.api._
import java.nio.{ByteBuffer, ByteOrder}

import org.platanios.tensorflow.api.core.types.DataType

import scala.collection.mutable

object BenchmarkArrayFeed {
  def array_feed(): Unit = {
    val ru = scala.reflect.runtime.universe

    // get runtime mirror
    val rm = ru.runtimeMirror(getClass.getClassLoader)

    // get companion method symbol
    val fromHostNativeSymbol = ru.typeOf[Tensor.type].decl(ru.TermName("fromHostNativeHandle")).asMethod
    //val fromHostNativeSymbol = ru.typeOf[DataType.type].decl(ru.TermName("fromHostNativeHandle")).asMethod

    // get method mirror
    val fromHostNativeMirror: ru.MethodMirror = rm.reflect(Tensor).reflectMethod(fromHostNativeSymbol)

    val num_obs = 1024
    // use method once?
    val initialize = NativeTensor.allocate(1, Array[Long](num_obs, 81, 4), num_obs * 81 * 4 * 4)
    val blank = NativeTensor.buffer(initialize).order(ByteOrder.nativeOrder)

    val feat_start_time = Calendar.getInstance().getTimeInMillis
    val array = Array.ofDim[Float](324)

    // dtype 1 == FLOAT32, https://github.com/eaplatanios/tensorflow_scala/blob/17f2247a1f590975eb069e82381bbd1417e85d00/modules/api/src/test/scala/org/platanios/tensorflow/api/core/DataTypeSpec.scala
    val hostHandle = NativeTensor.allocate(1, Array[Long](num_obs, 81, 4), num_obs * 81 * 4 * 4)
    val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)
    buffer.putFloat(10)

    val tensor: Tensor[Float] = fromHostNativeMirror(hostHandle).asInstanceOf[Tensor[Float]]
    val tensor_input = tensor.reshape(Shape(num_obs, 9, 9, 4))
    println("Feat Time %d".format(Calendar.getInstance().getTimeInMillis - feat_start_time))

    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")
    val eval_start_time = Calendar.getInstance().getTimeInMillis
    model.predict(tensor_input)

    println("Eval Time %d".format(Calendar.getInstance().getTimeInMillis - eval_start_time))
    //val eval_start_time = Calendar.getInstance().getTimeInMillis

    //while ( {
    // true
    //}) Thread.sleep(10000)
  }
  def static_feed(): Unit = {
    val model = new SimpleModel("/models/v4_15_policy.pb", "input_tensor:0", "output_softmax/Softmax:0")

    val tensor_input = Tensor.zeros[Float](Shape(10000, 9, 9, 4))
    val eval_start_time = Calendar.getInstance().getTimeInMillis
    model.predict(tensor_input)
    println("Eval Time %d".format(Calendar.getInstance().getTimeInMillis - eval_start_time))
  }
  def float_array_convert(): Unit = {
    val copy_start_time = Calendar.getInstance().getTimeInMillis
    val num_obs = 10000
    val hostHandle = NativeTensor.allocate(1, Array[Long](num_obs, 81, 4), num_obs * 81 * 4 * 4)
    val buffer = NativeTensor.buffer(hostHandle).order(ByteOrder.nativeOrder)

    val byte_array = new Array[Byte](4 * 9 * 9 * 4)
    for (_ <- 0 until num_obs) {
      buffer.put(byte_array)
    }

    println("Copy Time %d".format(Calendar.getInstance().getTimeInMillis - copy_start_time))
  }
  def main(args: Array[String]): Unit = {
    //float_array_convert()
    array_feed()
  }
}

