package learner

import breeze.linalg.DenseVector
import core.{BoardTransform, GameState, MoveSeq}

import scala.collection.mutable

object SearchNode {
  // Q-constants
  val MAX_Q: Double = 1.0
  // largest Q allowed by model prediction (MAX_Q is a minimax certified win)
  val MAX_MODEL_Q: Double = 1.0 - 1E-4
  val MIN_Q: Double = -MAX_Q
  val MIN_MODEL_Q: Double = -MAX_MODEL_Q
  val UNASSIGNED_Q: DenseVector[Double] = DenseVector(Double.NaN, Double.NaN, Double.NaN)
  val UNASSIGNED_P: Double = Double.NaN

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
    SearchNode(None, None, is_maximizing, params)
  }

}
class SearchNode private (parent: Option[SearchNode] = None, move: Option[Int], val is_maximizing: Boolean, params: Map[String, Any]) {
  // log of probability of playing this move given root board state (used for PVS search)
  var log_total_p: Double = parent match {
    case Some(`parent`) => SearchNode.UNASSIGNED_P
    case None => 0
  }

  private var _principal_variation: SearchNode = this

  private var _self_q: DenseVector[Double] = SearchNode.UNASSIGNED_Q

  private val _parents: mutable.Set[SearchNode] = parent.map(mutable.Set[SearchNode](_)) getOrElse mutable.Set[SearchNode]()

  val _full_move_list: MoveSeq = parent.map(_._full_move_list) getOrElse MoveSeq.empty()
  val _is_maximizing: Boolean = parent.map(!_._is_maximizing) getOrElse is_maximizing

  // is the game over with this move list
  private var _game_state: GameState = GameState.NOT_OVER

  // key is move index (just an integer)
  private val _children = Map[Int, SearchNode]()

  private val _children_with_q = mutable.Set[SearchNode]()

  // current best child
  private var _best_child: Option[SearchNode] = None

  private var _move_goodness: Double = 0.0

  def add_parent(parent: SearchNode, move: Int): Unit = {
    _parents += parent
    parent._children(move) = this
    if (assigned_q())
      parent._children_with_q += this
  }

  def create_child(move: Int): SearchNode = {
    SearchNode(this, move, !is_maximizing, params)
  }

  def add_child(child: SearchNode, move: Int): Unit = child.add_parent(this, move)

  def has_children(): Boolean = _children.nonEmpty
  def has_child(move: Int): Boolean = _children.contains(move)
  def get_child(move: Int): SearchNode = _children(move)
  def get_children(): Iterator[SearchNode] = _children.valuesIterator
  def has_parents(): Boolean = _parents.nonEmpty

  def pv(): SearchNode = _principal_variation
  def move_goodness(): Double = _move_goodness
  def game_state(): GameState = _game_state

  def get_q(): DenseVector[Double] = {
    if (has_nonself_pv()) _principal_variation.get_q()
    _self_q
  }

  def has_nonself_pv(): Boolean = _principal_variation != this
  def get_q(q_index: Int): Double = _principal_variation.get_q()(q_index)
  def assigned_q(): Boolean = _principal_variation._self_q != SearchNode.UNASSIGNED_Q || has_nonself_pv()
  def pv_depth(): Int = _principal_variation._full_move_list.moves.size

  def assign_p(p: Double): Unit = {
    assert (log_total_p == SearchNode.UNASSIGNED_P)
    log_total_p = _parents.head.log_total_p + p
  }

  // returns the move that moves us to the current state from the parent
  def get_move_relationship(parent: SearchNode): Int = {
    assert (parent.has_children())
    for ((move, child) <- parent._children) {
      if (this == child)
        return move
    }
    throw IndexOutOfBoundsException
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
    assert (!assigned_q())
    assert (q.size == 3)
    _self_q = q
    _update_q(this)
  }

  // should only be called after the q is directly updated
  def _update_q(pv: SearchNode): Unit = {
    assert (SearchNode.MIN_Q <= get_q()(SearchNode.DRAW_INDEX))
    for (parent <- _parents) {
      parent._children_with_q += this
    }
    _compute_move_goodness()
  }

  // used to evaluate how good a particular node is
  def _compute_move_goodness(): Unit = {
    val q_value = params("q_draw_vector").asInstanceOf[DenseVector[Double]].dot(get_q())

    // if there's a hard game-end conclusion, we want to pick shorter wins and longer losses
    if (_principal_variation._game_state.game_over) {
      val length_penalty = SearchNode.EPSILON * pv_depth() * (get_q(SearchNode.P1_WIN_INDEX) - get_q(SearchNode.P2_WIN_INDEX))
      _move_goodness = q_value + length_penalty
    }
    q_value
  }
  def _update_pv(): Unit = {
    // we are going to have to frequently search for the max, so might as well sort while we're at it
    if (is_maximizing)
      _best_child = Option[SearchNode](_children_with_q.maxBy(_._move_goodness))
    else
      _best_child = Option[SearchNode](_children_with_q.minBy(_._move_goodness))

    // our new PV is our best child's pv, or might just be best child if it doesn't have a pv
    _principal_variation = _best_child.map(_._principal_variation) getOrElse _best_child
  }

  def _is_better_than(other: SearchNode): Boolean = {
    is_maximizing && _move_goodness > other._move_goodness
  }

  // a bottom-up recomputation of pv based on q, should ideally only be called when there's been a change to the pv
  def update_pv(): Unit ={
    assert (has_children())
    _update_pv()
    for (parent <- _parents) {
      // if we're guaranteed to update the pv
      // 1. there is no pv from the parent
      // 2. this child is the pv
      // 3. this child should be the pv but isn't right now
      if (parent._best_child.isEmpty || parent._best_child.contains(this) || parent._best_child.exists(_is_better_than)) {
        update_pv()
      }
    }
  }
  /*
        # this should only be called on a parent node
        assert len(self.children_with_q) > 0

        self.update_best_child()

        self._update_pv(self.best_child.get_principal_variation())

        # update parents
        for parent in self._parents:
            # if this node is in the PV line
            if parent.best_child == self or parent.best_child is None or (
                    (parent.is_maximizing and self.better_q(parent.best_child)) or
                    (not parent.is_maximizing and not self.better_q(parent.best_child))
            ):
                parent.recalculate_q()
   */

  // because with transpositions, it is not guaranteed that the pv's move ordering is the most likely one
  def calculate_pv_order(): List[Int] = {
    val chain = mutable.ListBuffer[Int]()
    var current = this
    while (current.has_parents()) {
      var largest_p = Double.NegativeInfinity
      var best_move = 0
      var best_parent: SearchNode = current._parents.head
      for (parent <- current._parents) {
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

    val coord_moves = if (_full_move_list.moves.nonEmpty)
      calculate_pv_order().map(transformer.move_to_coordinate(_).toString).mkString(", ")
    else "(ROOT)"

    def assemble_string(pv: SearchNode): String = {
      f"PV: $coord_moves Q*: $pv.move_goodness P: $pv.log_total_p $pv.height%.4f"
    }
    assemble_string(_principal_variation)
  }
}
