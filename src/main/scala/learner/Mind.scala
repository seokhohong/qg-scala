package learner

import java.util.Calendar

import core._
import breeze.linalg.{*, DenseMatrix, DenseVector, clip, Transpose}
import breeze.numerics._
import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor}
import org.platanios.tensorflow.api.tensors.ops.Basic

import scala.util.control.Breaks._
import scala.collection.mutable

class SearchTree(board: Bitboard,
                 policy_est: SimpleModel,
                 value_est: SimpleModel,
                 private val search_params: Map[String, Any],
                 verbose: Boolean = true,
                 validation: Boolean = false) {
  private val is_maximizing = if (board.get_player_to_move() == Player.FIRST) true else false

  private val _search_params = initial_compute_of_params(search_params)

  val root_node: SearchNode = SearchNode.make_root(is_maximizing, _search_params)
  private val transformer = new BoardTransform(board.size)

  private var expandable_nodes = mutable.Set[SearchNode](root_node)
  private var _top_all_p = List[SearchNode](root_node)

  private var expansion_history = Set[SearchNode](root_node)

  private val transpositionTable = new TranspositionTable[SearchNode]()
  private val feature_board = new FeatureBoard(board)

  private val _size2 = board.size * board.size


  def initial_compute_of_params(params_base: Map[String, Any]): Map[String, Any] = {
    val draw_weight = params_base("q_draw_weight").asInstanceOf[Double]
    params_base + ("q_draw_vector" -> DenseVector[Double](1.0 - draw_weight, draw_weight, draw_weight - 1))
  }

  def get_pv(): SearchNode = root_node.pv()

  def add_expand_nodes(nodes : Set[SearchNode]): Unit = {
    expandable_nodes ++= nodes
    if (validation) {
      for (node <- nodes) {
        assert(!expansion_history.contains(node))
        assert(!node.game_over())
      }
      expansion_history ++= nodes
    }
  }

  // prefers pv's that have been explored more deeply
  private def confidence_score(node: SearchNode): Double = {
    val confidence_function = _search_params("confidence_function").asInstanceOf[Int => Double]
    val sign = if(node.is_maximizing) 1 else -1
    node.move_goodness() + confidence_function(node.pv_depth()) * sign
  }


  def compute_confidence_pv(): SearchNode = {
    val move_seq = MoveSeq.empty()

    var node = root_node
    breakable {
      while (node.has_nonself_pv()) {
        val children_with_q = node.get_children().filter(_.assigned_q())
        val best_child = if (node.is_maximizing) children_with_q.maxBy(confidence_score) else children_with_q.minBy(confidence_score)
        move_seq.append(best_child.get_move_relationship(node))
        node = best_child
      }
    }
    node
  }

  def add_expand_node(node: SearchNode): Unit ={
    expandable_nodes += node
    if (validation)
      assert(!expansion_history.contains(node))
  }

  def transposition_hits(): Int = transpositionTable.hits

  def remove_expand_nodes(parents: Set[SearchNode]): Unit = {
    expandable_nodes --= parents
  }
  // makes no call to models, just an object creation
  def create_child(parent: SearchNode, move: Int): Unit = {
    val child_transposition_hash = parent.get_move_list().hash_with_append(move)

    val lookup_child: Option[SearchNode] = transpositionTable.get_from_position_hash(child_transposition_hash)

    if (lookup_child.isDefined) {
      lookup_child.foreach(_.add_parent(parent, move))
      // if the child has q score, then update the parent to take it into consideration
      lookup_child.filter(_.assigned_q()).foreach(_ => parent.update_pv())
    } else {
      val child = parent.create_child(move)
      transpositionTable.put(child.get_move_list(), child)

      // if this is a winning state, mark it
      if (board.is_winning_move(move)) {
        board.make_move(move)
        child.assign_hard_q(board.game_state)
        board.unmove()

        // we would normally update q value here, but we do it outside this function
      } else {
        // normal expansion
        add_expand_node(child)
      }
    }
  }

  def create_children(parent: SearchNode, prediction_set: Transpose[DenseVector[Double]]): Unit = {

    // children we won't need to expand (i.e. there's a winning move from this position)
    val nonexpanding_children = mutable.Set[SearchNode]()

    // move forward
    val before_state = board.get_available_moves()
    parent.get_move_list().moves.foreach(board.make_move)
    breakable {
      for (move <- board.get_available_moves()) {
        // this works because move is an index nicely corresponding to the dense vector index
        val p_score = prediction_set(move)
        if (p_score >= _search_params("min_child_p").asInstanceOf[Double]) {
          create_child(parent, move)
          val child = parent.get_child(move)
          if (!child.assigned_p())
            child.assign_p(p_score)
          // if this is a guaranteed cutoff
          if (child.is_perfect_move()) {
            parent.update_pv()
            // then we shouldn't be expanding anything from this parent node
            nonexpanding_children ++= parent.get_children()
            break
          }
        }
      }
    }
    // reset
    parent.get_move_list().moves.foreach(_ => board.unmove())
    assert (board.get_available_moves() == before_state)
    remove_expand_nodes(nonexpanding_children.toSet)
  }

  def compute_p(parents: Set[SearchNode]): Unit = {
    if (verbose) println(f"Compute P ${parents.size}%d")

    val p_features = NativeTensorWrapper.allocate_floats(parents.size * _size2 * FeatureBoard.CHANNELS)

    val parents_list = parents.toList
    for (node <- parents_list) {
      node.get_move_list().moves.foreach(feature_board.make_move)
      feature_board.write_p_features_inplace(p_features)
      node.get_move_list().moves.foreach(_ => feature_board.unmove())
    }
    val predictions = policy_est.predict_to_matrix(p_features)

    //clip and log
    val log_predictions = predictions.mapValues { x =>
      math.log(math.max(math.min(x, 1), -1))
    }

    for (i <- 0 until predictions.rows) {
      create_children(parents_list(i), log_predictions(i, ::))
    }

    remove_expand_nodes(parents)
  }

  def update_top_all_p(new_nodes: Set[SearchNode]): Unit = {
    val num_pv_expand = _search_params("num_pv_expand").asInstanceOf[Int]
    _top_all_p ++= new_nodes
    _top_all_p = _top_all_p.sortBy(_.log_total_p).reverse.distinct.slice(0, num_pv_expand)
    if (validation) {
      assert(_top_all_p.contains(root_node))
    }
  }

  // Returns a list of top principal variations to expand
  def pv_expansion(to_p_expand: Set[SearchNode]): Set[SearchNode] = {
    if (to_p_expand.isEmpty) return Set[SearchNode]()

    update_top_all_p(to_p_expand)

    val top_pvs = mutable.Set[SearchNode]()
    for (node <- _top_all_p) {
      val pv = node.pv()
      // if the pv does not point at a game end state
      // there will be some overlap with to_p_expand
      if (!pv.game_over()) {
        top_pvs += pv
      }
    }

    if (validation) {
      assert(top_pvs.contains(root_node.pv()))
    }

    if (verbose) println(f"PV Expansion: ${top_pvs.size}%d")
    top_pvs.toSet
  }

  def highest_p(k: Int): Set[SearchNode] = {
    expandable_nodes.toList.sortBy(_.log_total_p).reverse.slice(0, k).toSet
  }

  def p_expand(): Unit = {
    val highest = highest_p(k = _search_params("p_batch_size").asInstanceOf[Int])
    val top_pvs = pv_expansion(highest)
    compute_p(highest ++ top_pvs)
  }

  def compute_q(candidates: Set[SearchNode]): Unit = {
    val no_q_candidates =  candidates.filter(!_.assigned_q())
    val all_parents = no_q_candidates.flatMap(_.parents)

    // pred_q_nodes are all the nodes that will have a q value predicted from the model
    val pred_q_nodes = mutable.ListBuffer[SearchNode]()

    // deal with the nodes with hard q
    for (node <- no_q_candidates) {
      node.get_move_list().moves.foreach(board.make_move)
      if (board.game_over())
        node.assign_hard_q(board.game_state)
      else
        pred_q_nodes += node
      node.get_move_list().moves.foreach(_ => board.unmove())
    }

    println(f"Computing Q: ${pred_q_nodes.size}%d")
    if (pred_q_nodes.isEmpty) return

    // make the q predictions
    val q_features = NativeTensorWrapper.allocate_floats(pred_q_nodes.size * _size2 * FeatureBoard.CHANNELS)
    for (node <- pred_q_nodes) {
      node.get_move_list().moves.foreach(feature_board.make_move)
      feature_board.write_p_features_inplace(q_features)
      node.get_move_list().moves.foreach(_ => feature_board.unmove())
    }

    val predictions = value_est.predict_to_matrix(q_features)

    assert (predictions.rows == pred_q_nodes.size)
    for (i <- 0 until predictions.rows) {
      val q_vector = predictions(i, ::).t.map(_.toDouble)
      pred_q_nodes(i).assign_leaf_q(q_vector / breeze.linalg.sum(q_vector))
    }

    all_parents.foreach(_.update_pv())
  }

  // this is necessary for learning data, regardless of root pv or whatnot
  private def make_root_q(): Unit = {
    val q_features = NativeTensorWrapper.allocate_floats(_size2 * FeatureBoard.CHANNELS)
    //val q_features = Tensor[Float](feature_board.get_q_features())
    feature_board.write_p_features_inplace(q_features)
    val prediction = value_est.predict_to_matrix(q_features)
    root_node.assign_leaf_q(prediction.toDenseVector.map(_.toDouble))
  }

  // Looks at top pv_expansion nodes by P value, taking each of their principal variations
  // With each pv, we find the highest likelihood child
  private def pv_q_expansion(): Set[SearchNode] = {
    val pv_q = mutable.Set[SearchNode]()
    for (node <- _top_all_p) {
      val pv = node.pv()
      if (pv.has_children()) {
        pv_q ++= pv.get_children().map(_.deepest_unexplored())
      }
    }
    println(f"PV Q Expansion ${pv_q.size}%d")
    pv_q.toSet
  }

  private def q_eval(): Unit = {
    val top_p = highest_p((_search_params("p_batch_size").asInstanceOf[Int] * _search_params("fraction_q").asInstanceOf[Double]).toInt)

    val top_pvs = pv_q_expansion()

    val to_eval = top_pvs ++ top_p

    if (to_eval.nonEmpty) {
      compute_q(to_eval)
    }
  }

  def run_validations(): Unit = {
    if (validation) {
      validate_proper_pv()
    }
  }

  def validate_proper_pv(): Unit = {
    for (node <- _top_all_p) {
      val best_move = node.best_move() getOrElse 0
      assert (node.pv().get_move_list().moves.contains(best_move))
      for (child <- node.pv().get_children()) {
        // this means we're failing to update pv somewhere important
        if (child.assigned_q()) {
          print("Wow")
        }
        assert (!child.assigned_q())
      }
    }
  }

  def run_iteration(): Unit = {
    if (expandable_nodes.isEmpty) return

    val orig_pv = root_node.pv()
    p_expand()

    // pv must have been expanded or changed
    if (validation) assert (root_node.pv().has_children() || root_node.pv() != orig_pv)

    if (verbose) {
      println(f"Num Leaf Nodes ${expandable_nodes.size}%d, Transposition ${transpositionTable.hits}%d")
    }

    q_eval()

    // pv must change
    if (validation) assert (root_node.pv() != orig_pv)

    run_validations()

    if (!root_node.has_self_q()) {
      make_root_q()
    }

    println(root_node)
  }

  def run_iterations(num_iterations: Int): Unit = {
    for (i <- 0 until num_iterations) {
      run_iteration()
    }
  }

  def run_time(custom_duration: Option[Double] = None): Unit = {
    val duration: Double = custom_duration getOrElse _search_params("max_thinktime").asInstanceOf[Double]
    val start_time = Calendar.getInstance().getTimeInMillis
    // convert duration to milliseconds

    while ((Calendar.getInstance().getTimeInMillis - start_time < duration * 1000) && expandable_nodes.nonEmpty) {
      run_iteration()
    }
  }
}

class Mind(model_file: String, search_params: Map[String, Any], verbose: Boolean = true, validate: Boolean = false) {

  val (value_est, policy_est) = load_models()

  def load_models(): (SimpleModel, SimpleModel) = {
    (new SimpleModel(f"${model_file}_value.pb", input_name="input_tensor:0", output_name="output_softmax/Softmax:0"),
            new SimpleModel(f"${model_file}_policy.pb", input_name="input_tensor:0", output_name="output_softmax/Softmax:0")
      )
  }

  def make_search(board: Bitboard): SearchTree = {
    new SearchTree(board, policy_est, value_est, search_params, verbose, validate)
  }

  def make_move(board: Bitboard, searcher: Option[SearchTree] = None): SearchNode = {
    val search = searcher getOrElse make_search(board)
    search.run_time()
    search.root_node
  }

  // avoid anonymous function?
  def make_move_debug(board: Bitboard): SearchNode = {
    val search = make_search(board)
    search.run_time()
    search.root_node
  }
}
