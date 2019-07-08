package core

import java.io.{BufferedInputStream, FileInputStream}
import breeze.linalg.DenseVector

import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.core.client.{FeedMap, Session}
import org.tensorflow.framework.GraphDef
import org.platanios.tensorflow.api.{Graph, Output, Shape, Tensor}

class SimpleModel(model_file: String, input_name: String, output_name: String) {
  private val graphDef = GraphDef.parseFrom(
    new BufferedInputStream(new FileInputStream(getClass.getResource(model_file).getFile)))
  private val graph = Graph.fromGraphDef(graphDef)
  private val session = Session(graph)
  private val input_tensor = graph.getOutputByName(input_name)
  private val output_tensor = graph.getOutputByName(output_name)

  def predict(input_values: Tensor[Any]): Seq[DenseVector[Float]] = {
    val feedsMap: FeedMap = FeedMap(Map(input_tensor -> input_values.asInstanceOf[Tensor[Any]]))

    val result_seq: Seq[Tensor[Any]] = session.run(fetches = Seq(output_tensor), feeds = feedsMap)

    result_seq.map(_.entriesIterator.map(_.asInstanceOf[Float]).toArray).map(new DenseVector[Float](_))
  }
}
