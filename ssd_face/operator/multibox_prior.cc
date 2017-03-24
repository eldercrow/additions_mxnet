/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_prior.cc
 * \brief generate multibox prior boxes cpu implementation
 * \author Joshua Zhang
*/

#include "./multibox_prior-inl.h"

namespace mshadow {
template<typename DType>
inline void MultiBoxPriorForward(const Tensor<cpu, 2, DType> &out,
                            const std::vector<float> &sizes,
                            const std::vector<float> &ratios,
                            const int in_width, const int in_height) {
  const float step_x = 1.f / in_width;
  const float step_y = 1.f / in_height;
  const int num_sizes = static_cast<int>(sizes.size());
  const int num_ratios = static_cast<int>(ratios.size());
  int count = 0;

  for (int r = 0; r < in_height; ++r) {
    float center_y = (r + 0.5) * step_y;
    for (int c = 0; c < in_width; ++c) {
      float center_x = (c + 0.5) * step_x;
      // ratio = 1, various sizes
      for (int i = 0; i < num_sizes; ++i) {
        float size = sizes[i];
        float w = size / 2;
        float h = size / 2;
        out[count][0] = center_x - w;  // xmin
        out[count][1] = center_y - h;  // ymin
        out[count][2] = center_x + w;  // xmax
        out[count][3] = center_y + h;  // ymax
        ++count;
      }
      // various ratios, size = min_size = size[0]
      float size = sizes[0];
      for (int j = 1; j < num_ratios; ++j) {
        float ratio = sqrtf(ratios[j]);
        float w = size * ratio / 2;
        float h = size / ratio / 2;
        out[count][0] = center_x - w;  // xmin
        out[count][1] = center_y - h;  // ymin
        out[count][2] = center_x + w;  // xmax
        out[count][3] = center_y + h;  // ymax
        ++count;
      }
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(MultiBoxPriorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxPriorOp<cpu, DType>(param);
  });
  return op;
}

Operator* MultiBoxPriorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiBoxPriorParam);

MXNET_REGISTER_OP_PROPERTY(MultiBoxPrior, MultiBoxPriorProp)
.add_argument("data", "Symbol", "Input data.")
.add_arguments(MultiBoxPriorParam::__FIELDS__())
.describe("Generate prior(anchor) boxes from data, sizes and ratios.");

}  // namespace op
}  // namespace mxnet
