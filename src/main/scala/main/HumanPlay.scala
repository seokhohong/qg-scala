package main

import core.{Bitboard, BoardTransform, Player}
import learner.Mind

import scala.util.control.Breaks._
import scala.util.Random

object HumanPlay {
  def opponent(): Mind = {
    new Mind("/models/v4_15", Map[String, Any](
      "min_child_p" -> -7.0,
      "p_batch_size" -> (1 << 10),
      "fraction_q" -> 1.0,
      "q_draw_weight" -> 0.0,
      "max_thinktime" -> 20.0,
      "num_pv_expand" -> 60
    ))
  }
  def play(): Unit = {
    val board = new Bitboard(size=9, win_chain_length=5)
    val transformer = new BoardTransform(9)
    val mind: Mind = opponent()
    val human_player = Player.FIRST

    val size = 9

    for (i <- 0 until Random.nextInt(10))
      board.make_random_move()

    println(board)

    while(true) {
      breakable {
        if (board.get_player_to_move() == human_player) {
          print("Input your move (i.e. \"3 5\"): ")
          val inp = scala.io.StdIn.readLine()
          if (inp.split(" ") != 2)
            println("Incorrect number of coordinates, please try again!")
            break

          val coords = inp.split(' ')
          val x = coords(0).toInt
          val y = coords(1).toInt
          if (x < 0 || x >= size || y < 0 || y >= size) {
            println("Coordinate Out of bounds!")
            break
          }
          val index = transformer.coordinate_to_move(x, y)
          if (!board.move_available(index)) {
            println("Invalid Move!")
            break
          }
          board.make_move(index)
        }
        else {
          println("Computer is Thinking...")
          println(board.move_history)
          val search_root = mind.make_move(board)
          val move = search_root.get_move_list().moves.head
          board.make_move(move)
          println("Move: ", move, "Q: ", search_root.self_q(), "Search Q: ", search_root.get_q())

        }
        print(board)
      }
      if (board.game_over()) {
        if (board.get_winning_player == human_player) {
          println("YOU WIN!")
        }
        else if (board.get_winning_player == human_player.other()) {
          print("COMPUTER WINS!")
        }
        else {
          print("DRAW!")
        }
      }
    }
  }
  def main(args: Array[String]): Unit = {
    play()

/*
    # randomize the board a bit
    for j in range(random.randint(0, int(SIZE * 2))):
        board.make_random_move()

    print(board.pprint())

    while True:
        if board.get_player_to_move() == Player.SECOND:
            inp = input("Input your move (i.e. \"3 5\"): ")
            if len(inp.split(' ')) != 2:
                print('Incorrect number of coordinates, please try again!')
                continue
            x, y = inp.split(' ')
            try:
                x = int(x)
                y = int(y)
            except:
                print('Please input Numbers!')
                continue
            if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
                print('Out of bounds!')
                continue
            index = board._transformer.coordinate_to_index(x, y)
            if not board.is_move_available(index):
                print('Invalid Move!')
                continue
            result = board.move_coord(x, y)
            print(board)
        else:
            print('Computer is thinking...')
            print(board._ops)

            move, current_q, best_q = mind.make_move(board)
            print(" ")
            print(move, 'Q:', best_q)

            # if best_q > PExpNode.MAX_MODEL_Q :
            #    print('Computer Resigns!')
            #    break

            board.move(move)
            print(board)

        if board.game_over():
            if board.get_winning_player() == Player.FIRST:
                print('COMPUTER WINS!')
            elif board.get_winning_player() == Player.SECOND:
                print('YOU WIN!')
            else:
                print("DRAW!")
            break

 */
  }
}