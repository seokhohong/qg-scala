package core

import java.io.{BufferedInputStream, FileInputStream}
import breeze.linalg.{DenseVector, DenseMatrix}

import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.core.client.{FeedMap, Session}
import org.tensorflow.framework.GraphDef
import org.platanios.tensorflow.api.{Graph, Output, Shape, Tensor}
import scala.reflect.ClassTag
import org.platanios.tensorflow.api.tensors.ops.Basic

class SimpleModel(model_file: String, input_name: String, output_name: String) {
  private val graphDef = GraphDef.parseFrom(
    new BufferedInputStream(new FileInputStream(getClass.getResource(model_file).getFile)))
  private val graph = Graph.fromGraphDef(graphDef)
  private val session = Session(graph)
  private val input_tensor = graph.getOutputByName(input_name)
  private val output_tensor = graph.getOutputByName(output_name)

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
}
