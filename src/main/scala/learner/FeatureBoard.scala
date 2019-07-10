package learner

import collection.mutable
import core._
import breeze.linalg._
import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor, UINT8}

object FeatureBoard {
  val CHANNELS: Int = 4
}
class FeatureBoard(board: core.Bitboard) {
  private val _size = board.size
  private val size2 = _size * _size
  private val _move_history = mutable.ListBuffer[Int](board.move_history: _*)
  private val _original_move_length = _move_history.size

  private val board_array = DenseMatrix.zeros[Int](_size * _size, FeatureBoard.CHANNELS)
  val _transformer = new BoardTransform(_size)
  private var _player_to_move = board.get_player_to_move()

  private var depth = 0

  _init_board_array(board)

  def _init_board_array(board: Bitboard): Unit = {
    for (i <- 0 until size2) {
      if (board.get_spot(i) == Player.FIRST)
        board_array(i, 0) = 1
      else if (board.get_spot(i) == Player.SECOND)
        board_array(i, 1) = 1
    }
    if (_move_history.nonEmpty) {
      _set_last_move(_move_history.last)
    }
    _update_last_player()
  }

  def player_to_move(): Player = _player_to_move

  def _init_available_move_vector(): Unit = {

  }

  def _set_last_move(last_move: Int): Unit = {
    board_array(last_move, 2) = 1
  }

  def _clear_last_move(last_move: Int): Unit = {
    board_array(last_move, 2) = 0
  }

  def get_features(): DenseMatrix[Int] = {
     board_array.copy
  }
  def get_p_features(): DenseMatrix[Int] = get_features()
  def get_q_features(): DenseMatrix[Int] = get_features()

  def _update_last_player(): Unit = {
    val last_player_val = if (_player_to_move == Player.FIRST) -1 else 1
    board_array(::, 3) := last_player_val
  }

  def _set_spot(move: Int): Unit = {
    if (_player_to_move == Player.FIRST)
      board_array(move, 0) = 1
    else
      board_array(move, 1) = 1
  }
  def _clear_spot(move: Int): Unit = {
    board_array(move, 0) = 0
    board_array(move, 1) = 0
  }
  def make_move(move: Int): Unit = {
    depth += 1
    if (_move_history.nonEmpty)
      _clear_last_move(_move_history.last)

    _move_history += move
    _set_spot(move)
    _set_last_move(move)
    _player_to_move = _player_to_move.other()
    _update_last_player()
  }
  def unmove(): Unit = {
    depth -= 1
    val prev_last_move = _move_history.last
    _move_history -= prev_last_move
    _clear_last_move(prev_last_move)
    _clear_spot(prev_last_move)
    if (_move_history.nonEmpty)
      _set_last_move(_move_history.last)
    _player_to_move = _player_to_move.other()
    _update_last_player()
  }
  // whether the board is in the reset state
  def is_reset(): Boolean = depth == 0
}
