#include "./ternarize-inl.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(TernarizeParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new TernarizeOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet


