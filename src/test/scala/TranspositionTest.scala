import core.{MoveSeq, TranspositionTable}
import org.scalatest.FunSuite

class TranspositionTest extends FunSuite {
  test(testName = "simple") {
    val table = new TranspositionTable[Int]()
    table.put(MoveSeq.empty(), 10)
    assert(table.get(MoveSeq.empty()).contains(10))
    assert(table.hits == 1)

    val m1 = MoveSeq.empty().append(10).append(15).append(20)
    val m2 = MoveSeq.empty().append(20).append(15).append(10)
    val m3 = MoveSeq.empty().append(10).append(20).append(15)

    assert (m1.hashCode == m2.hashCode)
    // test move transposition
    table.put(m1, 6)
    assert(table.get(m2).contains(6))
    assert(table.get(m3).isEmpty)
  }
  test(testName = "not hit") {
    val table = new TranspositionTable[Int]()
    table.put(MoveSeq.empty(), 10)
    table.get(MoveSeq.empty().append(10))
    assert(table.hits == 0)
  }
}
