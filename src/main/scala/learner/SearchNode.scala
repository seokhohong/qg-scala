package learner

import breeze.linalg.DenseVector
import core.{BoardTransform, GameState, MoveSeq}

import scala.collection.mutable
import scala.math

object SearchNode {
  // Q-constants
  val MAX_Q: Double = 1.0
  // largest Q allowed by model prediction (MAX_Q is a minimax certified win)
  val MAX_MODEL_Q: Double = 1.0 - 1E-4
  val MIN_Q: Double = -MAX_Q
  val MIN_MODEL_Q: Double = -MAX_MODEL_Q
  val LARGE_VALUE: Double = 10
  val UNASSIGNED_Q: DenseVector[Double] = DenseVector(-LARGE_VALUE, -LARGE_VALUE, -LARGE_VALUE)
  val UNASSIGNED_P: Double = LARGE_VALUE

  //dimension of Q vector
  val Q_DIM: Int = 3

  // Q index, since q-values are a tuple
  val P1_WIN_INDEX = 0
  val DRAW_INDEX = 1
  val P2_WIN_INDEX = 2

  val P1_WIN_Q: DenseVector[Double] = DenseVector[Double](1, 0, 0)
  val DRAW_Q: DenseVector[Double] = DenseVector[Double](0, 1, 0)
  val P2_WIN_Q: DenseVector[Double] = DenseVector[Double](0, 0, 1)

  val EPSILON = 1E-6

  /*
    Optional Params:
      "q-draw-weight": higher draw weight means preferring positions of higher draw likelihood, all things being equal
   */
  def make_root(is_maximizing: Boolean, params: Map[String, Any]): SearchNode = {
    new SearchNode(None, None, is_maximizing, params)
  }

}
class SearchNode private (parent: Option[SearchNode] = None, move: Option[Int], val is_maximizing: Boolean, params: Map[String, Any]) {
  // log of probability of playing this move given root board state (used for PVS search)
  var log_total_p: Double = parent.map(_ => SearchNode.UNASSIGNED_P) getOrElse 0

  private var _principal_variation: SearchNode = this

  private var _self_q: DenseVector[Double] = SearchNode.UNASSIGNED_Q

  val parents: mutable.Set[SearchNode] = parent.map(mutable.Set[SearchNode](_)) getOrElse mutable.Set[SearchNode]()

  // this is an example of options being really terrible
  private val base_list = parent.map(_.get_move_list()) getOrElse MoveSeq.empty()
  private val _full_move_list = move.map(base_list.append) getOrElse base_list

  private val _is_maximizing: Boolean = parent.map(!_._is_maximizing) getOrElse is_maximizing

  // is the game over with this move list
  private var _game_state: GameState = GameState.NOT_OVER

  // key is move index (just an integer)
  private val _children = mutable.Map[Int, SearchNode]()

  private val _children_with_q = mutable.ListBuffer[SearchNode]()

  // current best child
  private var _best_child: Option[SearchNode] = None

  private var _move_goodness: Double = 0.0

  def add_parent(parent: SearchNode, move: Int): Unit = {
    assert (parent != this)
    parents += parent
    parent._children(move) = this
    if (assigned_q())
      if (!parent._children_with_q.contains(this))
        parent._children_with_q += this
  }

  def create_child(move: Int): SearchNode = {
    val new_child = new SearchNode(Some(this), Some(move), !is_maximizing, params)
    add_child(new_child, move)
    new_child
  }

  def add_child(child: SearchNode, move: Int): Unit = child.add_parent(this, move)

  def has_children(): Boolean = _children.nonEmpty
  def has_child(move: Int): Boolean = _children.contains(move)
  def get_child(move: Int): SearchNode = _children(move)
  def get_children(): Iterator[SearchNode] = _children.valuesIterator
  def has_parents(): Boolean = parents.nonEmpty
  def get_move_list(): MoveSeq = _full_move_list

  def pv(): SearchNode = _principal_variation
  def best_move(): Option[Int] = _best_child.map(_.get_move_list().moves.head)
  def move_goodness(): Double = _move_goodness
  def game_state(): GameState = _game_state
  def game_over(): Boolean = _game_state.game_over
  def has_self_q(): Boolean = _self_q != SearchNode.UNASSIGNED_Q
  def self_q(): DenseVector[Double] = _self_q

  // returns if we don't need to search for a better move
  def is_perfect_move(): Boolean =  (_game_state == GameState.WIN_PLAYER_1 && get_q() == SearchNode.P1_WIN_Q) ||
                                    (_game_state == GameState.WIN_PLAYER_2 && get_q() == SearchNode.P2_WIN_Q)

  def get_q(): DenseVector[Double] = {
    if (has_nonself_pv())
      return _principal_variation.get_q()
    _self_q
  }

  def has_nonself_pv(): Boolean = _principal_variation != this
  def get_q(q_index: Int): Double = _principal_variation.get_q()(q_index)
  def assigned_q(): Boolean = has_self_q() || has_nonself_pv()
  def assigned_p(): Boolean = log_total_p != SearchNode.UNASSIGNED_P
  def pv_depth(): Int = _principal_variation._full_move_list.moves.size

  def assign_p(p: Double): Unit = {
    //assert (log_total_p == SearchNode.UNASSIGNED_P)
    if (parents.isEmpty) {
      throw new IllegalArgumentException("Cannot assign p to a root node")
    }
    log_total_p = parents.head.log_total_p + p
  }

  // returns the move that moves us to the current state from the parent
  def get_move_relationship(parent: SearchNode): Int = {
    assert (parent.has_children())
    for ((move, child) <- parent._children) {
      if (this == child)
        return move
    }
    throw new AssertionError("get_move_relationship was called on a parent child pair that does not exist")
  }

  // when we're assigning a final game state
  def assign_hard_q(new_game_state: GameState): Unit = {
    _game_state = new_game_state
    val q = _game_state match {
      case GameState.WIN_PLAYER_1 => SearchNode.P1_WIN_Q
      case GameState.WIN_PLAYER_2 => SearchNode.P2_WIN_Q
      case GameState.DRAW => SearchNode.DRAW_Q
    }
    assign_leaf_q(q)
  }

  // assigning q value to a leaf node
  def assign_leaf_q(q: DenseVector[Double]): Unit = {
    //is there a way to check this without compiler optimizing this out?
    //assert (!assigned_q())
    assert (q.size == 3)
    _self_q = q
    _update_q(this)
  }

  // should only be called after the q is directly updated
  private def _update_q(pv: SearchNode): Unit = {
    assert (SearchNode.MIN_Q <= get_q()(SearchNode.DRAW_INDEX))
    for (parent <- parents) {
      if (!parent._children_with_q.contains(this))
        parent._children_with_q += this
    }
    _compute_move_goodness()
  }

  // used to evaluate how good a particular node is
  private def _compute_move_goodness(): Unit = {
    val q_value = params("q_draw_vector").asInstanceOf[DenseVector[Double]].dot(get_q())

    // if there's a hard game-end conclusion, we want to pick shorter wins and longer losses
    if (_principal_variation._game_state.game_over) {
      val length_penalty = SearchNode.EPSILON * pv_depth() * (get_q(SearchNode.P1_WIN_INDEX) - get_q(SearchNode.P2_WIN_INDEX))
      _move_goodness = q_value - length_penalty
    } else {
      _move_goodness = q_value
    }
  }

  // returns the leaf node with highest P
  // assumes that this node does not have explored children
  def deepest_unexplored(): SearchNode = {
    assert (!assigned_q())
    if (!has_children()) {
      return this
    }
    get_children().maxBy(_.log_total_p).deepest_unexplored()
  }

  private def _update_pv(): Unit = {
    assert (_children_with_q.nonEmpty)

    // we are going to have to frequently search for the max, so might as well sort while we're at it
    // right now we do just max or minsearch
    val best_child: SearchNode = if (is_maximizing) _children_with_q.maxBy(_._move_goodness) else _children_with_q.minBy(_._move_goodness)
    _best_child = Some(best_child)

    // our new PV is our best child's pv, or might just be ourselves if it doesn't have a pv
    // I believe we should always have a best_child at this point...
    assert (_best_child.isDefined)
    _principal_variation = _best_child.map(_.pv()) getOrElse this
    _compute_move_goodness()
  }

  // whether this node is "better" than another
  private def _is_better_than(other: SearchNode): Boolean = {
    is_maximizing && _move_goodness > other._move_goodness
  }

  // a bottom-up recomputation of pv based on q, should ideally only be called when there's been a change to the pv
  def update_pv(): Unit ={
    assert (has_children())
    _update_pv()
    for (parent <- parents) {
      // if we're guaranteed to update the pv
      // 1. there is no pv from the parent
      // 2. this child is the pv
      // 3. this child should be the pv but isn't right now
      if (parent._best_child.isEmpty || parent._best_child.contains(this) || parent._best_child.exists(_is_better_than)) {
        parent.update_pv()
      }
    }
  }

  class SearchQuality() {
    val children_counts: mutable.Map[SearchNode, Int] = mutable.Map[SearchNode, Int]()
    val scores: mutable.Map[SearchNode, Double] = mutable.Map[SearchNode, Double]()
    var composite_score: Double = 0

    compute()
    private def compute(): Unit = {
      count_children(SearchNode.this)

      // each node will have counted itself via this algorithm
      for (node <- children_counts.keys) {
        children_counts(node) -= 1
      }

      recurse_score_node(SearchNode.this)
      composite_score = scores.values.sum / scores.values.size
    }
    private def count_children(node: SearchNode): Unit = {
      for (child <- node.get_children()) {
        count_children(child)
      }
      children_counts(node) = node.get_children().map(children_counts(_)).sum + 1
    }
    private def recurse_score_node(node: SearchNode): Unit = {
      val children_with_pv = node.get_children().filter(_.has_nonself_pv())
      children_with_pv.foreach(recurse_score_node)

      if (node.has_nonself_pv()) {
        scores(node) = score_node(node)
      }
    }
    private def score_node(node: SearchNode): Double = {
      assert (node.has_nonself_pv())
      math.log((children_counts(node.pv) + 1).toDouble / children_counts(node))
    }
  }
  // a metric to define search goodness--it will help track improvements we make to the search algorithm
  def compute_search_goodness(): SearchQuality = {
    new SearchQuality()
  }

  // because with transpositions, it is not guaranteed that the pv's move ordering is the most likely one
  def calculate_pv_order(): List[Int] = {
    val chain = mutable.ListBuffer[Int]()
    var current = this
    while (current.has_parents()) {
      var largest_p = Double.NegativeInfinity
      var best_move = 0
      var best_parent: SearchNode = current.parents.head
      for (parent <- current.parents) {
        if (parent.log_total_p > largest_p) {
          best_move = current.get_move_relationship(parent)
          best_parent = parent
          largest_p = parent.log_total_p
        }
      }
      chain += best_move
      current = best_parent
    }
    chain.toList.reverse
  }

  override def toString: String = {
    val transformer = new BoardTransform(size=9)

    val coord_moves = if (has_nonself_pv() && pv()._full_move_list.moves.nonEmpty)
      pv().calculate_pv_order().map(transformer.move_to_coordinate(_).toString).mkString(", ")
    else if (parents.isEmpty) {
      "(ROOT)"
    }
    else {
      "(POS): %s".format(_full_move_list.moves.map(transformer.move_to_coordinate(_).toString).mkString(", "))
    }

    def assemble_string(pv: SearchNode): String = {
      f"PV: $coord_moves Q*: ${pv.move_goodness()}%1.4f P: ${pv.log_total_p}%1.4f Q: (${pv.get_q(SearchNode.P1_WIN_INDEX)}%1.4f" +
        f", ${pv.get_q(SearchNode.DRAW_INDEX)}, ${pv.get_q(SearchNode.P2_WIN_INDEX)})"
    }
    assemble_string(_principal_variation)
  }
}
