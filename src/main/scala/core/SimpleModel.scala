package core

import java.io.{BufferedInputStream, FileInputStream}

import breeze.linalg.{DenseMatrix, DenseVector}
import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.core.client.{FeedMap, Session}
import org.tensorflow.framework.GraphDef
import org.platanios.tensorflow.api.{Graph, Output, Shape, Tensor}
import org.platanios.tensorflow.jni.{Tensor => NativeTensor}

import scala.reflect.ClassTag
import org.platanios.tensorflow.api.tensors.ops.Basic
import java.nio.{ByteBuffer, ByteOrder}

import core.NativeTensorWrapper.{rm, ru}
import main.BenchmarkArrayFeed.getClass

object SimpleModel {
  // reflection access to fromHostNativeHandle private method
  private val ru = scala.reflect.runtime.universe

  // get runtime mirror
  private val rm = ru.runtimeMirror(getClass.getClassLoader)

  // get companion method symbol
  private val fromHostNativeSymbol = ru.typeOf[Tensor.type].decl(ru.TermName("fromHostNativeHandle")).asMethod
  //val fromHostNativeSymbol = ru.typeOf[DataType.type].decl(ru.TermName("fromHostNativeHandle")).asMethod

  // get method mirror
  private val fromHostNativeMirror: ru.MethodMirror = rm.reflect(Tensor).reflectMethod(fromHostNativeSymbol)


}
class SimpleModel(model_file: String, input_name: String, output_name: String, size: Int = 9, channels: Int = 4) {
  private val graphDef = GraphDef.parseFrom(
    new BufferedInputStream(new FileInputStream(getClass.getResource(model_file).getFile)))
  private val graph = Graph.fromGraphDef(graphDef)
  private val session = Session(graph)
  private val input_tensor = graph.getOutputByName(input_name)
  private val output_tensor = graph.getOutputByName(output_name)

  private val bytes_per_float = 4



  def predict_tensor(input_values: Tensor[Any]): Seq[Tensor[Any]] = {
    val feedsMap: FeedMap = FeedMap(Map(input_tensor -> input_values.asInstanceOf[Tensor[Any]]))

    session.run(fetches = Seq(output_tensor), feeds = feedsMap)
  }
  // TODO: Surely there's a better way of doing these transforms
  def predict(input_values: Tensor[Any]): DenseMatrix[Float] = {
    val result_seq = predict_tensor(input_values).head
    //val to_array = Basic.splitEvenly(result_seq.head, 0, result_seq.head.shape(0))
    val matrix = new DenseVector[Float](result_seq.entriesIterator.map(_.asInstanceOf[Float]).toArray)
    //matrix.asDenseMatrix.reshape(result_seq.shape(0), result_seq.shape(1))
    matrix.asDenseMatrix.reshape(result_seq.shape(1), result_seq.shape(0)).t
  }
  def _predict_buffer_to_tensor(input_buffer: NativeTensorWrapper): Tensor[Float] = {
    val tensor: Tensor[Float] = input_buffer.toFloatTensor

    val matrix = new DenseVector[Float](tensor.entriesIterator.toArray)

    val num_obs = input_buffer.floatsCapacity / (size * size * channels)
    val tensor_input = tensor.reshape(Shape(num_obs, size, size, channels))

    predict_tensor(tensor_input).head.asInstanceOf[Tensor[Float]]
  }
  /*
  def predict(input_buffer: NativeTensorWrapper): NativeTensorWrapper = {
    val tensor_output = _predict_buffer_to_tensor(input_buffer)
    NativeTensorWrapper.from_tensor(tensor_output)
  }
  def predict_to_matrix_unsafe(input_buffer: NativeTensorWrapper): DenseMatrix[Float] = {
    val tensor_output = _predict_buffer_to_tensor(input_buffer)
    predict(input_buffer).toMatrix(tensor_output.shape(1), tensor_output.shape(0))
  }
  def predict_to_array_unsafe(input_buffer: NativeTensorWrapper): Array[Float] = {
    predict(input_buffer).toArray
  }
   */
  def predict_to_array(input_buffer: NativeTensorWrapper): Array[Float] = {
    val tensor_output = _predict_buffer_to_tensor(input_buffer)
    tensor_output.entriesIterator.toArray
  }
  def predict_to_matrix(input_buffer: NativeTensorWrapper): DenseMatrix[Float] = {
    val tensor_output = _predict_buffer_to_tensor(input_buffer)
    val matrix = new DenseVector[Float](tensor_output.entriesIterator.toArray)
    //matrix.asDenseMatrix.reshape(result_seq.shape(0), result_seq.shape(1))
    matrix.asDenseMatrix.reshape(tensor_output.shape(1), tensor_output.shape(0)).t
  }
}
