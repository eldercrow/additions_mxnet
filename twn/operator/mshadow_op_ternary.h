#ifndef MXNET_OPERATOR_MSHADOW_OP_TERNARY_H_
#define MXNET_OPERATOR_MSHADOW_OP_TERNARY_H_

#include <mxnet/base.h>
#include "special_functions-inl.h"
#include "mshadow_op.h"

namespace mxnet {
namespace op {
namespace mshadow_op {

struct ternarize {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType th) {
    if(a < -th) return DType(-1);
    else if(a > th) return DType(1);
    else return DType(0);
  }
};

// struct qrelu {
//   template<typename DType>
//   MSHADOW_XINLINE static DType Map(DType a, DType th) {
//     if (a < DType(0.f)) return DType(0.f);
//     else if(a > th) return th;
//     else return a;
//   }
// };
//
// struct qrelu_grad {
//   template<typename DType>
//   MSHADOW_XINLINE static DType Map(DType a, DType th) {
//     return (a > DType(0.f) && a < th) ? DType(1.f) : DType(0.f);
//   }
// };
//
// struct qrelu_scale_round {
//   template<typename DType>
//   MSHADOW_XINLINE static DType Map(DType a, DType th, DType scale) {
//     if (a < DType(0.f))
//       return DType(0.f);
//     else if (a > th)
//       return roundf(th * scale) / scale;
//     else
//       return roundf(a * scale) / scale;
//   }
// };

}  // namespace mshadow_op
}  // namespace op
}  // namespace mxnet
#endif // MXNET_OPERATOR_MSHADOW_OP_H_
