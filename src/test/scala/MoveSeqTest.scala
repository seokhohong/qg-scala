import core.{SimpleModel, _}
import learner._
import org.platanios.tensorflow.api.{Shape, Tensor}
import org.scalatest.FunSuite

class MoveSeqTest extends FunSuite {
  test("move sequence transposition") {
    val m1 = MoveSeq.empty().append(15).append(23).append(13)
    val m2 = MoveSeq.empty().append(13).append(23).append(15)
    assert (m1.position_hash == m2.position_hash)
  }
}
