package core

import scala.collection.parallel.mutable

class TranspositionTable[V] {
  private val table = mutable.ParMap[List[Int], V]()
  var hits = 0
  def get(key: MoveSeq): Option[V] = {
    get_from_position_hash(key.position_hash)
  }
  def get_from_position_hash(position_hash: List[Int]): Option[V] = {
    hits += 1
    table.get(position_hash)
  }
  def two_val_function(key: MoveSeq, item: Int): Unit = {
    print("Hi")
  }
  def put(key: MoveSeq, item: V): Unit = {
    table.put(key.position_hash, item)
  }
}