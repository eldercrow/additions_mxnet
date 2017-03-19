#ifndef MXNET_OPERATOR_MSHADOW_OP_QUANTIZE_H_
#define MXNET_OPERATOR_MSHADOW_OP_QUANTIZE_H_

#include <mxnet/base.h>
#include "special_functions-inl.h"
#include "mshadow_op.h"

namespace mxnet {
namespace op {
namespace mshadow_op {

struct clamp_zero_max {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType th_max) {
    if (a < DType(0)) return DType(0);
    else if(a > th_max) return DType(th_max);
    return a;
  }
};

struct clamp_grad {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType th_min, DType th_max) {
    return (a > th_min && a < th_max) ? DType(1.f) : DType(0.f);
  }
};

// quantize a value whose range is [0, 1]
struct quantize_zero_one {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, float scale) {
    return static_cast<DType>(roundf(static_cast<float>(a) * scale));
  }
};

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif // MXNET_OPERATOR_MSHADOW_OP_QUANTIZE_H_
