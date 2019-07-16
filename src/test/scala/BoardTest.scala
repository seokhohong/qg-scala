package scala

import org.scalatest.FunSuite
import core._
import breeze.linalg._
import net.liftweb.json
import net.liftweb.json.JsonAST._
import net.liftweb.json.Extraction._

class BoardTest extends FunSuite {
  test("board.BoardTransform") {
    val rot = new BoardTransform(9)
    for (move <- rot.all_moves) {
      val (x, y) = rot.move_to_coordinate(move)
      assert(move == rot.coordinate_to_move(x, y))
    }
  }
  test(testName="bitboard checklocations"){
    val cache = new BitboardCache()

    assert(cache.get_check_locations(0, -1, 0) == List[Int]())
    assert(cache.get_check_locations(0, 1, 0) == List[Int](1, 2, 3, 4))
    assert(cache.get_check_locations(0, 0, 1) == List[Int](9, 18, 27, 36))
  }
  test(testName="test rotator") {
    val rot = new BoardTransform(size=9)

    val board = new Bitboard()

    val moves = List(0, 19, 50, 40)
    for (move <- moves) {
      board.make_move(move)
    }

  }
  test(testName = "simple win") {
    val board = new Bitboard(size=9, win_chain_length = 5)
    for (move <- List(0, 1, 9, 10, 18, 19, 27, 28)) {
      board.make_move(move)
      assert (board.game_state == GameState.NOT_OVER)
    }
    board.make_move(36)
    board.pprint()
    assert (board.game_state == GameState.WIN_PLAYER_1)
  }
  test(testName = "available_moves") {
    val board = new Bitboard(size=9, win_chain_length = 5)
    for (move <- List(0, 1, 9, 10, 18, 19, 27, 28)) {
      board.make_move(move)
    }
    assert (board.get_available_moves().contains(36))
    assert (!board.get_available_moves().contains(19))
  }
  test(testName = "move history construction") {
    val move_list = List(0, 1, 9, 10, 18, 19, 27, 28)
    val board = new Bitboard(size=9, win_chain_length = 5)
    for (move <- move_list) {
      board.make_move(move)
    }
    val board2 = new Bitboard(size=9, win_chain_length = 5, initial_moves = move_list)
    assert (board2.export() == board.export())
  }
  test(testName = "export test") {
    val board = new Bitboard(size = 9, win_chain_length = 5)
    for (move <- List(0, 1, 9, 10, 18, 19, 27, 28)) {
      board.make_move(move)
    }
    val recreated_board = Bitboard.create(board.export())
    assert (board.export() == recreated_board.export())
  }
  test(testName = "mass play") {
    val board = new Bitboard(size=7, win_chain_length = 4)
    //board.make_random_move()
  }
  test(testName = "rotation") {
    val transformer = new BoardTransform(size=9)
    assert (transformer.get_rotated_points(15) == List[Int](15, 55, 19, 11, 65, 25, 61, 69))
  }

  test(testName = "winning_move") {
    for (i <- 0 until 100) {
      val board = new Bitboard(size = 9, win_chain_length = 5)
      while (!board.game_over()) {
        board.make_random_move()
        val last_move = board.move_history.last
        if (board.game_state == GameState.WIN_PLAYER_1 || board.game_state == GameState.WIN_PLAYER_2) {
          board.unmove()
          board.make_move(last_move)
          assert (board.game_over())
        }
      }
    }
  }

  test(testName = "test draw") {
    val moves = List[Int](49, 40, 50, 29, 51, 52, 47, 48, 41, 31, 32, 23, 39, 68, 42, 58, 38, 33, 65, 56, 57, 73, 67, 37, 64, 66,
    43, 59, 21, 22, 13, 25, 24, 44, 30, 16, 3, 12, 2, 1, 5, 6, 14, 17, 35, 7, 71, 28, 19, 34, 60, 78, 77,
    62, 46, 54, 55, 36, 20, 45, 63, 18, 27, 53, 15, 9, 0, 69, 10, 75, 74, 8, 79, 72, 80)
    val board = new Bitboard(size = 9, win_chain_length = 5, initial_moves=moves)

    while (!board.game_over()) {
      board.make_random_move()
    }

    assert (board.game_over())
  }
  test(testName = "test undo win") {
    val moves = List[Int](0, 1, 9, 10, 18, 19, 27, 28)
    val board = new Bitboard(size = 9, win_chain_length = 5, initial_moves=moves)

    board.make_move(36)

    assert (board.game_over())

    board.unmove()

    assert (board.get_available_moves().size == 81 - 8)
    assert (!board.game_over())
  }
}