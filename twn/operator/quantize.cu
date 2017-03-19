#include "./quantize-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(QuantizeParam param) {
  Operator *op = NULL;
  op = new QuantizeOp<gpu>(param);
  return op;
}

}  // namespace op
}  // namespace mxnet



