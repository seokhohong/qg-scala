package core

import breeze.linalg._

import collection.mutable
import collection.immutable._
import util.bitops

import scala.util.Random
import net.liftweb.json
import net.liftweb.json.Extraction.decompose
import net.liftweb.json.JsonAST.compactRender

//TODO: Int should be a class, but not good enough at Scala yet

object Player{
  val FIRST = new Player("FIRST", 1)
  val SECOND = new Player("SECOND", 2)
  val NONE = new Player("NONE", 0)
}

class Player(name: String, val value: Int) {
  def other(): Player ={
    if (this == Player.FIRST) return Player.SECOND
    else if (this == Player.SECOND) return Player.FIRST
    Player.NONE
  }
  override def toString: String = name
}

object GameState{
  val NOT_OVER = new GameState("NOT_OVER", false, Player.NONE)
  val DRAW = new GameState("DRAW", true, Player.NONE)
  val WIN_PLAYER_1 = new GameState("WIN_PLAYER_1", true, Player.FIRST)
  val WIN_PLAYER_2 = new GameState("WIN_PLAYER_2", true, Player.SECOND)
}

class GameState(_name: String, _game_over: Boolean, _winning_player: Player){
  def game_over: Boolean = _game_over
  def winning_player: Player = _winning_player
  def name: String = _name
  override def toString: String = name
}

class BoardTransform(size: Int) {
  //make this immutable
  private val cached_point_rotations = mutable.HashMap[Int, List[Int]]()
  private val NUM_ROTATIONS = 8

  private val size2 = size * size
  private val _all_moves = 0 until size2

  cache_rotations()

  def all_moves: IndexedSeq[Int] = _all_moves

  private def cache_rotations(): Unit ={
    val position_matrix = DenseMatrix.zeros[Int](size, size)
    val point_rotation_builder = mutable.HashMap[Int, mutable.ListBuffer[Int]]()
    // fill matrix with their index value
    for (i <- 0 until size2) {
      point_rotation_builder(i) = mutable.ListBuffer[Int]()
    }
    for (x <- 0 until size) {
      for (y <- 0 until size) {
        position_matrix(x, y) = coordinate_to_move(x, y)
      }
    }

    for (matrix <- get_rotated_matrices_int(position_matrix)){
      for (x <- 0 until size) {
        for (y <- 0 until size) {
          point_rotation_builder(matrix(x, y)) += position_matrix(x, y)
        }
      }
    }
    for (move <- _all_moves) {
      cached_point_rotations(move) = point_rotation_builder(move).toList
    }
  }
  // stupid bug where I can't access elements of a densevector
  def get_rotated_matrices_int(matrix: DenseMatrix[Int]): List[DenseMatrix[Int]] = {
    assert(matrix.rows == size)

    List(
      matrix,
      matrix.t,
      rot90(matrix),
      rot90(matrix).t,
      rot90(matrix, 2),
      rot90(matrix, 2).t,
      rot90(matrix, 3),
      rot90(matrix, 3).t,
    )
  }

  def get_rotated_tensor(tensor: List[DenseMatrix[Int]]): List[List[DenseMatrix[Int]]] = {
    tensor.map(get_rotated_matrices_int).transpose
  }

  def get_rotated_points(point: Int): List[Int] ={
    cached_point_rotations(point)
  }

  def coordinate_to_move(x: Int, y: Int): Int ={
    x * size + y
  }

  def move_to_coordinate(index: Int): (Int, Int)={
    (index / size, index % size)
  }

  def in_bounds(x: Int, y: Int): Boolean={
    x >= 0 && x < size && y >= 0 && y < size
  }
}



object BitboardCache {
  val DELTAS: List[(Int, Int)] = List[(Int, Int)]((-1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0), (1, -1), (-1, 1))
  val DELTA_SETS: List[((Int, Int), (Int, Int))] = List(((-1, -1), (1, 1)), ((0, -1), (0, 1)), ((-1, 0), (1, 0)), ((1, -1), (-1, 1)))
}

class BitboardCache(size: Int = 9, win_chain_length: Int = 5) {
  private val size2 = size * size
  private val all_moves = 0 until size2

  private val _check_locations = mutable.HashMap[(Int, Int, Int), List[Int]]()
  private val _transformer = new BoardTransform(size)

  build_check_locations()

  // bitset is mutable since the board's bitset is mutable and we have to combine
  private val _delta_masks = mutable.HashMap[(Int, Int), mutable.BitSet]()
  private val _win_checks = mutable.HashMap[(Int, Int), List[BitSet]]()
  build_win_checks()

  // define the list of squares that need to be checked for a win condition
  def build_check_locations(): Unit = {
    for (delta_pair <- BitboardCache.DELTAS) {
      for (move <- all_moves) {
        var coords = mutable.ListBuffer[Int]()
        val (center_x, center_y) = _transformer.move_to_coordinate(move)
        for (j <- 1 until win_chain_length) {
          val step_x = delta_pair._1 * j
          val step_y = delta_pair._2 * j
          if (_transformer.in_bounds(center_x + step_x, center_y + step_y)) {
            coords += _transformer.coordinate_to_move(center_x + step_x, center_y + step_y)
          }
        }
        _check_locations((move, delta_pair._2, delta_pair._1)) = coords.toList
      }
    }
  }
  // which bits have to be checked for a win condition given the index and deltasets
  private def _marked_locations(move: Int, deltaset_index: Int): List[Int] = {
    val deltaset = BitboardCache.DELTA_SETS(deltaset_index)
    _check_locations((move, deltaset._1._1, deltaset._1._2)) ++
      _check_locations((move, deltaset._2._1, deltaset._2._2)).sorted
  }
  /*
      def _marked_locations(self, index, deltas):
        board = Board(size=self._size, win_chain_length=self._win_chain_length)
        all_locations = []
        for delta in deltas:
            all_locations.extend(board._check_locations[(index, *delta)])
        return sorted(all_locations)
   */

  def get_check_locations(move: Int, delta_x: Int, delta_y: Int): List[Int]={
    _check_locations((move, delta_x, delta_y))
  }

  def check_win(bitset: mutable.BitSet, move: Int): Boolean = {
    for (i <- 0 until 4) {
      if (_win_checks((move, i)).contains(bitset & _delta_masks((move, i)))) {
        return true
      }
    }
    false
  }

  private def build_win_checks(): Unit ={
    for (move <- all_moves){
      for (deltaset_index <- BitboardCache.DELTA_SETS.indices) {
        val marked_locations = _marked_locations(move, deltaset_index)
        _delta_masks((move, deltaset_index)) = mutable.BitSet(marked_locations: _*)
        val win_list = mutable.ListBuffer[BitSet]()
        for (i <- 0 to marked_locations.size - (win_chain_length - 1)) {
          win_list += BitSet(marked_locations.slice(i, i + win_chain_length - 1): _*)
        }
        _win_checks((move, deltaset_index)) = win_list.toList
      }
    }

  }
}

case class BoardImport(size: Int, win_chain_length: Int, move_history: List[Int])

object Bitboard {
  // loading a Bitboard from json string
  def create(json_string: String): Bitboard = {
    implicit val formats = net.liftweb.json.DefaultFormats
    val board_import = json.parse(json_string).extract[BoardImport]
    val board = new Bitboard(size=board_import.size,
                          win_chain_length=board_import.win_chain_length,
                          initial_moves=board_import.move_history)
    board
  }
  /*
      def create_board_from_specs(BitBoard, cache, size, win_chain_length, boardstring):
        board = BitBoard(cache=cache, size=size, win_chain_length=win_chain_length)
        player_1_moves = []
        player_2_moves = []
        reshaped_board = np.array([int(elem) for elem in boardstring]).reshape((size, size))
        for i in range(size):
            for j in range(size):
                index = board._transformer.coordinate_to_index(i, j)
                if reshaped_board[i][j] == Player.FIRST.value:
                    player_1_moves.append(index)
                elif reshaped_board[i][j] == Player.SECOND.value:
                    player_2_moves.append(index)

        for i in range(len(player_1_moves)):
            board.blind_move(player_1_moves[i])
            if i < len(player_2_moves):
                board.blind_move(player_2_moves[i])

        return board
   */
}

class Bitboard(val size: Int = 9,
               val win_chain_length: Int = 5,
               initial_moves: List[Int] = List[Int]()) {
  private val cache = new BitboardCache(size, win_chain_length)
  private val transformer = new BoardTransform(size)

  private var _game_state = GameState.NOT_OVER

  private val size2 = size * size

  private val _move_history = mutable.ListBuffer[Int]()

  private val full_bitset = BitSet((0 until size2): _*)
  private val bitsets = List[mutable.BitSet](new mutable.BitSet(), new mutable.BitSet())

  _init(initial_moves)

  def _init(initial_moves: List[Int]) {
    for (move <- initial_moves) {
      make_move(move)
    }
  }

  def get_spot(move: Int): Player = {
    if (bitsets.head.contains(move)) {
      return Player.FIRST
    } else if (bitsets(1).contains(move)) {
      return Player.SECOND
    }
    Player.NONE
  }

  def get_spot_coord(x: Int, y: Int): Player = {
    get_spot(transformer.coordinate_to_move(x, y))
  }

  def get_player_to_move(): Player = {
    if (_move_history.size % 2 == 0) {
      return Player.FIRST
    }
    Player.SECOND
  }

  private def bitset_of_player_to_move(): mutable.BitSet = {
    bitset_of_player(get_player_to_move())
  }
  private def bitset_of_player(player: Player): mutable.BitSet = {
    if (player == Player.FIRST) {
      return bitsets.head
    }
    bitsets(1)
  }

  def blind_move(move: Int): Unit = {
    bitset_of_player_to_move() += move
    _move_history += move
  }

  def unmove(): Unit = {
    val last_move = _move_history.last
    bitset_of_player_to_move() -= last_move
    _move_history -= last_move
    _game_state = GameState.NOT_OVER
  }

  def make_move(move: Int): Unit ={
    assert (_game_state == GameState.NOT_OVER)
    blind_move(move)
    _check_game_over(move)
  }

  //call after making a move
  def _check_game_over(move: Int): Unit ={
    val last_player = get_player_to_move().other()
    if (cache.check_win(bitset_of_player(last_player), move)) {
      if (get_player_to_move() == Player.SECOND) {
        _game_state = GameState.WIN_PLAYER_1
      }
      else {
        _game_state = GameState.WIN_PLAYER_2
      }
    }
    if (_move_history.size == size2) _game_state = GameState.DRAW
  }

  def game_over(): Boolean = {
    game_state != GameState.NOT_OVER
  }

  def game_state: GameState = _game_state
  def get_winning_player: Player = _game_state.winning_player
  def move_history: List[Int] = _move_history.toList

  def get_available_moves(): BitSet = {
    full_bitset ^ (bitsets.head | bitsets(1))
  }

  def move_available(index: Int): Boolean = {
    get_available_moves().contains(index)
  }

  def make_random_move(): Unit = {
    val available_moves = get_available_moves().toList
    assert (available_moves.nonEmpty)
    make_move(available_moves(Random.nextInt(available_moves.size)))
  }

  def export(): String = {
    var boardstring = ""
    for (x <- 0 until size) {
      for (y <- 0 until size) {
        boardstring += get_spot(transformer.coordinate_to_move(x, y)).value.toString
      }
    }
    val map = HashMap(
      "size" -> size,
      "win_chain_length" -> win_chain_length,
      "move_history" -> _move_history
    )

    implicit val formats = net.liftweb.json.DefaultFormats
    compactRender(decompose(map))
  }

  def _display_char(move: Int, last_move_highlight: Boolean): Char = {
    if (_move_history.nonEmpty) {
      val last_move = _move_history.last
      val was_last_move = last_move == move
      if (bitsets.head.contains(move)) {
        if (was_last_move && last_move_highlight) {
          return 'X'
        }
        return 'x'
      }
      else if (bitsets(1).contains(move)) {
        if (was_last_move && last_move_highlight) {
          return 'O'
        }
        return 'o'
      }
    }
    ' '
  }

  def pprint(last_move_highlight: Boolean = true): String = {
    var board_string = " "
    for (i <- 0 until size) {
      board_string += " " + i.toString
    }
    for (i <- 0 until size) {
      board_string += "\n" + i.toString
      for (j <- 0 until size) {
        board_string += "|" + _display_char(transformer.coordinate_to_move(j, i), last_move_highlight)
      }
      board_string += "|"
    }
    board_string
  }
}