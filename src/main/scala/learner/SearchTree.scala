package learner

import core._
import breeze.linalg.{DenseVector, clip}
import breeze.numerics._
import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor}
import org.platanios.tensorflow.api.tensors.ops.Basic

import scala.collection.mutable

class SearchTree(board: Bitboard,
                 policy_est: SimpleModel,
                 value_est: SimpleModel,
                 private val search_params: Map[String, Any],
                 verbose: Boolean = true,
                 validation: Boolean = false) {
  private val is_maximizing = if (board.get_player_to_move() == Player.FIRST) true else false

  private val root_node = SearchNode.make_root(is_maximizing, search_params)
  private val transformer = new BoardTransform(board.size)

  private var expandable_nodes = Set[SearchNode](root_node)
  private var _top_all_p = List[SearchNode](root_node)

  private var expansion_history = Set[SearchNode](root_node)

  private val transpositionTable = new TranspositionTable[SearchNode]()
  private val feature_board = new FeatureBoard(board)

  private val _search_params = initial_compute_of_params(search_params)

  def initial_compute_of_params(params_base: Map[String, Any]): Map[String, Any] = {
    val draw_weight = params_base("q_draw_weight").asInstanceOf[Int]
    params_base + ("q_draw_vector" -> DenseVector[Double](1.0 - draw_weight, draw_weight, draw_weight - 1))
  }

  def get_pv(): SearchNode = root_node.pv()

  def add_expand_nodes(nodes : Set[SearchNode]): Unit = {
    expandable_nodes ++= nodes
    if (validation) {
      for (node <- nodes) {
        assert(!node.game_over())
      }
      expansion_history ++= nodes
    }
  }

  def remove_expand_nodes(parents: Set[SearchNode]): Unit = {
    expandable_nodes --= parents
  }
  // makes no call to models, just a
  def create_child(parent: SearchNode, move: Int): Unit = {
    val child_transposition_hash = parent.get_move_list().hash_with_append(move)

    val lookup_child: Option[SearchNode] = transpositionTable.get_from_position_hash(child_transposition_hash)

    val to_update = mutable.Set[SearchNode]()
    val to_expand = mutable.Set[SearchNode]()

    if (lookup_child.isDefined) {
      lookup_child.foreach(_.add_parent(parent, move))
    } else {
      val child = parent.create_child(move)
      transpositionTable.put(child.get_move_list(), child)

      // if this is a winning state, mark it
      if (board.is_winning_move(move)) {
        board.make_move(move)
        child.assign_hard_q(board.game_state)
        board.unmove()

        // since children don't automatically update their q/pv's, we aggregate the parents and do it all together
        to_update += parent
      } else {
        // normal expansion
        to_expand += child
      }

    }
    // update pv's of each parent, since we didn't do that
    to_update.foreach(_.update_pv())
    // add to the global list of nodes to expand
    add_expand_nodes(to_expand.toSet)

  }

  def compute_p(parents: Set[SearchNode]): Unit = {
    if (verbose) println("Compute P", parents.size)

    val p_features = mutable.ListBuffer[Tensor[Float]]()
    for (node <- parents) {
      node.get_move_list().moves.foreach(feature_board.make_move)
      p_features += feature_board.get_p_features().toArray
      node.get_move_list().moves.foreach(_ => feature_board.unmove())
    }
    val predictions: Seq[DenseVector[Float]] = policy_est.predict(Basic.stack(p_features).reshape(Shape(p_features.size, board.size, board.size, FeatureBoard.CHANNELS)))

    predictions.foreach(log(_))
    //predictions.map(scala.math.max(_.asInstanceOf[Int], 0.0.asInstanceOf[Int]))

    //val clipped = logged.
    for ((prediction_set, parent) <- predictions zip parents) {
      val child_creation_vector = prediction_set
      // move forward
      parent.get_move_list().moves.foreach(board.make_move)
      for (move <- (0 until 81)) {
        create_child(parent, move)
        val child = parent.get_child(move)
        if (!child.assigned_p())
          child.assign_p(5)
      }
      // reset
      parent.get_move_list().moves.foreach(_ => board.unmove())
    }
    remove_expand_nodes(parents)
  }
/*
    def compute_p(self, parents):
        if self._verbose:
            print('Compute P', len(parents))
        p_features = []

        for node in parents:
            if self._validations:
                assert node.log_total_p != PExpNodeV4.UNASSIGNED_P
                assert node.game_status == GameState_v2.NOT_OVER
            node_p_features = self.thought_board.get_p_features_after(node.get_move_chain())
            p_features.append(node_p_features)


        p_features_tensor = np.array(p_features).reshape((-1, self.board.get_size(), self.board.get_size(), FeatureBoard_v2.CHANNELS))

        # returns (len(parents), size**2) matrix
        predictions = self.policy_est.predict(p_features_tensor, batch_size=len(p_features))
        log_p_predictions = np.log(np.clip(predictions, a_min=keras.backend.epsilon(), a_max=1.))

        # create and assign
        for prediction_set, parent in zip(log_p_predictions, parents):
            child_creation_vector = np.logical_and(prediction_set > self.min_child_p,
                                                   self.thought_board.get_available_move_vector_after(
                                                       parent.get_move_chain()))
            # prepare thoughtboard for child creation
            self.thought_board.make_moves(parent.get_move_chain())
            for move in child_creation_vector.nonzero()[0]:
                child = self.create_child(parent, int(move))
                # may have already been assigned p if we're using transposition table
                if not child.is_assigned_p():
                    child.assign_p(prediction_set[move])

            self.thought_board.reset()
            # remove the parents from possible expanding nodes
            self.remove_expand_node(parent)
 */
  // Returns a list of top principal variations to expand
  def pv_expansion(to_p_expand: Set[SearchNode]): Set[SearchNode] = {
    if (to_p_expand.isEmpty) return Set[SearchNode]()

    _top_all_p ++= to_p_expand.slice(0, _search_params("num_pv_expand").asInstanceOf[Int])

    val top_pvs = mutable.Set[SearchNode]()
    for (node <- _top_all_p) {
      val pv = node.pv()
      // node has a pv, which is not already counted, and is not already going to be expanded
      // and if the pv does not point at a game end state
      if (node.has_nonself_pv() && !to_p_expand.contains(pv) && !pv.game_over()) {
        top_pvs += node
      }
    }
    if (verbose) println(f"PV Expansion: ${top_pvs.size}%d")
    top_pvs
  }

  def highest_p(k: Int): Set[SearchNode] = {
    expandable_nodes.toList.sortBy(_.log_total_p).toSet
  }

  def p_expand(): Set[SearchNode] = {
    val highest = highest_p(k = _search_params("p_batch_size").asInstanceOf[Int])
    val top_pvs = pv_expansion(highest)
    compute_p(highest ++ top_pvs)
    top_pvs
  }

  def compute_q(candidates: Set[SearchNode]): Unit = {
    val no_q_candidates =  candidates.filter(!_.assigned_q())
    val all_parents = no_q_candidates.flatMap(_.parents)
    for (node <- no_q_candidates) {
      node.get_move_list().moves.foreach(board.make_move)
      if (board.game_over()) {

      }
      node.get_move_list().moves.foreach(_ => board.unmove)
    }
  }
/*
    def compute_q(self, candidates):
        # we compute only for nodes that haven't finished the game
        q_features = []
        # recalculate qs only on parents
        parents = set()
        original_candidates = len(candidates)
        assert len(candidates) == len(set(candidates))
        candidates = [node for node in candidates if not node.is_assigned_q()]

        q_nodes = []
        for node in candidates:
            parents.update(node.get_parents())
            self.thought_board.make_moves(node.get_move_chain())
            if not self.thought_board.game_over():
                q_features.append(self.thought_board.get_q_features())
                q_nodes.append(node)
            else:
                node.assign_hard_q(self.thought_board.get_game_status())
            self.thought_board.reset()

        print('Candidate Q', original_candidates, 'Compute Q', len(q_features))

        # if we don't have any q's to expand after hard_q assessment
        if len(q_features) == 0:
            return

        q_feature_tensor = np.array(q_features).reshape((-1, self.board.get_size(), self.board.get_size(), FeatureBoard_v2.CHANNELS))

        q_predictions = np.clip(self.value_est.predict(q_feature_tensor, batch_size=len(q_features)).reshape((-1, 3)),
                                a_min=PExpNodeV4.MIN_MODEL_Q, a_max=PExpNodeV4.MAX_MODEL_Q)

        assert len(q_predictions) == len(q_nodes)
        for i, node in enumerate(q_nodes):
            node.assign_leaf_q(q_predictions[i])

        for parent in parents:
            parent.recalculate_q()
  */

}
