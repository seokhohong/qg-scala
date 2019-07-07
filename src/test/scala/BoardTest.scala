import org.scalatest.FunSuite
import core._

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
      board.move(move)
    }
  }
  test(testName = "simple win") {
    val board = new Bitboard(size=9, win_chain_length = 5)
    for (move <- List(0, 1, 9, 10, 18, 19, 27, 28)) {
      board.move(move)
    }
    board.move(36)
    board.pprint()
    assert (board.game_state() == GameState.WIN_PLAYER_1)
  }
  test(testName = "mass play") {
    val board = new Bitboard(size=7, win_chain_length = 4)
    //board.make_random_move()
  }
  /*
      def test_mass_play(self):
        for i in range(1000):
            board = Board(size=7, win_chain_length=4)
            board.make_random_move()
            if board.game_over():
                self.assertTrue(board.game_won() or board.game_assume_drawn())
            if board.game_assume_drawn():
                self.assertTrue(board.game_over())
      def test_rotator(self):
        rot = BoardTransform(size=9)

        board = Board(size=9, win_chain_length=5)

        moves = [(0, 0), (2, 1), (5, 5), (4, 4)]
        for i, move in enumerate(moves):
            board.move_coord(move[0], move[1])
            print(board.pprint())
            index = rot.coordinate_to_index(move[0], move[1])
            rotated_matrices = rot.get_rotated_matrices(board._matrix.reshape((board.get_size(), board.get_size(), 1)))
            self.assertFalse(np.equal(rotated_matrices[0], rotated_matrices[2]).all())
            for point, mat in zip(rot.get_rotated_points(index), rotated_matrices):
                x, y = rot.index_to_coordinate(point)
                self.assertEqual(mat[x][y][0], board.get_player_last_move().value)
   */
}