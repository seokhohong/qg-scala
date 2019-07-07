package learner

import collection.mutable
import core._
import breeze.linalg._
import org.platanios.tensorflow.api.{Graph, Session, Shape, Tensor, UINT8}

class FeatureBoard(board: core.Bitboard) {
  val CHANNELS = 4
  private val _size = board.size
  private val size2 = _size * _size
  val _move_history = mutable.ListBuffer[Int](board.move_history: _*)
  val _original_move_length = _move_history.size

  val board_array = DenseMatrix.zeros[Int](_size * _size, CHANNELS)
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
    /*
    # this is prematurely optimized
    # input is a single integer
    def move(self, move):
        assert move is not None
        # convert integer to x, y

        last_move = self._ops[-1] if len(self._ops) > 0 else None

        # clear last move
        if last_move is not None:
            self.tensor[last_move, 2] = 0

        # make the move
        self._ops.append(move)

        # set spot
        player_index = 0 if self._player_to_move == Player.FIRST else 1
        self.tensor[move, player_index] = 1

        # set last move
        self.tensor[move, 2] = 1

        # flip which player moves next
        self._player_to_move = self._player_to_move.other

        # fill features of last player to move
        player_value = -1 if self._player_to_move == Player.FIRST else 1
        self.tensor[:, 3].fill(player_value)

    def unmove(self):
        last_move = self._ops.pop()
        self._clear_last_move(last_move)
        self._clear_spot(last_move)
        self._update_last_move(self._last_move())
        self._player_to_move = self._player_to_move.other
        self._update_last_player()

   */
}
/*
 def __init__(self, board):
        self._size = board.get_size()
        self._ops = board.get_move_history()
        self._original_move_length = len(self._ops)
        self.tensor = np.zeros((self._size ** 2, FeatureBoard_v2.CHANNELS), dtype=np.int8)
        self._player_to_move = board.get_player_to_move()
        self._transformer = BoardTransform(self._size)
        self._init_available_move_vector(board)
        self._init(board)
        self._init_last_move(board)

    def _init_last_move(self, board):
        last_move = board.get_last_move()
        if last_move is not None:
            self._update_last_move(last_move)

    def _init_available_move_vector(self, board):
        self._available_move_vector = np.zeros((self._size ** 2))
        for i in range(self._size ** 2):
            if board.is_move_available(i):
                self._available_move_vector[i] = 1

    # returns the available move vector at the board's initial state
    def get_init_available_move_vector(self):
        return np.copy(self._available_move_vector)

 */