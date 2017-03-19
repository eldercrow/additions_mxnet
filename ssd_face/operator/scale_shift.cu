/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm.cu
 * \brief
 * \author Bing Xu
*/

#include "./scale_shift-inl.h"
// #include "./cudnn_scale_shift-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(ScaleShiftParam param, int dtype) {
  return new ScaleShiftOp<gpu>(param);
}

}  // namespace op
}  // namespace mxnet


