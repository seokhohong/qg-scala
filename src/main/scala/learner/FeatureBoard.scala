package learner

import java.nio.ByteBuffer

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

  //private val board_array = DenseMatrix.zeros[Int](_size * _size, FeatureBoard.CHANNELS)
  //private val board_array = Tensor.zeros[Int](Shape(size2, FeatureBoard.CHANNELS))
  private val board_array = ByteBuffer.wrap(new Array[Byte](size2 * FeatureBoard.CHANNELS * 4))
  val _transformer = new BoardTransform(_size)
  private var _player_to_move = board.get_player_to_move()

  private var depth = 0

  _init_board_array(board)

  def _set_elem(index: Int, channel: Int, value: Float): Unit = {
    val byte_index = (index * FeatureBoard.CHANNELS + channel) * 4
    board_array.putFloat(byte_index, value)
  }


  def _init_board_array(board: Bitboard): Unit = {
    for (i <- 0 until size2) {
      if (board.get_spot(i) == Player.FIRST)
        _set_elem(i, 0, 1)
      else if (board.get_spot(i) == Player.SECOND)
        _set_elem(i, 1, 1)
    }
    if (_move_history.nonEmpty) {
      _set_last_move(_move_history.last)
    }
    _update_last_player()
  }

  def player_to_move(): Player = _player_to_move

  def _set_last_move(last_move: Int): Unit = {
    _set_elem(last_move, 2, 1)
  }

  def _clear_last_move(last_move: Int): Unit = {
    _set_elem(last_move, 2, 0)
  }

  def get_features_as_array(): Array[Float] = {
    val float_array = new Array[Float](size2 * FeatureBoard.CHANNELS)
    board_array.asFloatBuffer().get(float_array)
    float_array
  }
  def get_features_as_nice_array(): Array[Array[Array[Float]]] = {
    val flat_array: Array[Float] = get_features_as_array()
    val nice_array = Array.ofDim[Float](_size, _size, FeatureBoard.CHANNELS)
    for (i <- 0 until _size) {
      for (j <- 0 until _size) {
        for (k <- 0 until FeatureBoard.CHANNELS) {
          nice_array(i)(j)(k) = flat_array(i * 9 * 4 + j * 4 + k)
        }
      }
    }
    nice_array
  }
  // premature optimization
  def write_features_inplace(buffer: ByteBuffer): Unit = {
    buffer.put(board_array)
  }
  def write_p_features_inplace(buffer: ByteBuffer): Unit = write_features_inplace(board_array)
  def write_q_features_inplace(buffer: ByteBuffer): Unit = write_features_inplace(board_array)

  def _update_last_player(): Unit = {
    val last_player_val = if (_player_to_move == Player.FIRST) -1 else 1
    for (i <- 0 until size2) {
      _set_elem(i, 3, last_player_val)
    }
  }

  def _set_spot(move: Int): Unit = {
    if (_player_to_move == Player.FIRST)
      _set_elem(move, 0, 1)
    else
      _set_elem(move, 1, 1)
  }
  def _clear_spot(move: Int): Unit = {
    _set_elem(move, 0, 0)
    _set_elem(move, 0, 0)
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
  def multi_move(moves: List[Int]): Unit = {
    if (moves.isEmpty) return

    depth += moves.size

    if (_move_history.nonEmpty)
      _clear_last_move(_move_history.last)

    _move_history ++= moves
    moves.foreach(_set_spot)
    _set_last_move(moves(-1))

    _player_to_move = _player_to_move.other()
    _update_last_player()
  }

  def multi_unmove(count: Int): Unit = {
    if (count == 0) return
    depth -= count

    //for (move <- _move_history(-count )) {
    //  board_array(move) = 0
    //}
  }
  // whether the board is in the reset state
  def is_reset(): Boolean = depth == 0
}
