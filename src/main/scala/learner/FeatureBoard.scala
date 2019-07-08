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
  val _move_history = mutable.ListBuffer[Int](board.move_history: _*)
  val _original_move_length = _move_history.size

  val board_array = DenseMatrix.zeros[Int](_size * _size, FeatureBoard.CHANNELS)
  val _transformer = new BoardTransform(_size)
  var _player_to_move = board.get_player_to_move()

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
      _update_last_player()
    }
  }


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
    if (_player_to_move == Player.FIRST)
      board_array(::, 3) := 1
    else
      board_array(::, 3) := -1
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
    if (_move_history.nonEmpty)
      _clear_last_move(_move_history.last)

    _move_history += move
    _set_spot(move)
    _set_last_move(move)
    _update_last_player()
    _player_to_move = _player_to_move.other()
  }
  def unmove(): Unit = {
    val prev_last_move = _move_history.last
    _move_history -= prev_last_move
    _clear_last_move(prev_last_move)
    _clear_spot(prev_last_move)
    _set_last_move(_move_history.last)
    _player_to_move = _player_to_move.other()
    _update_last_player()
  }
}
