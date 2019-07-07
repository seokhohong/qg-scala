package core

import java.io.{BufferedInputStream, FileInputStream}

import org.platanios.tensorflow.api.Graph
import org.platanios.tensorflow.api.core.client.{FeedMap, Session}
import org.tensorflow.framework.GraphDef
import org.platanios.tensorflow.api.{Graph, Output, Shape, Tensor}

class SimpleModel(model_file: String, input_name: String, output_name: String) {
  val graphDef = GraphDef.parseFrom(
    new BufferedInputStream(new FileInputStream(getClass.getResource(model_file).getFile)))
  val graph = Graph.fromGraphDef(graphDef)
  val session = Session(graph)
  val input_tensor = graph.getOutputByName(input_name)
  val output_tensor = graph.getOutputByName(output_name)

  def predict(input_values: Tensor[Any]): Seq[Tensor[Any]] = {
    val feedsMap: FeedMap = FeedMap(Map(input_tensor -> input_values.asInstanceOf[Tensor[Any]]))

    session.run(fetches = Seq(output_tensor), feeds = feedsMap)
  }
}
