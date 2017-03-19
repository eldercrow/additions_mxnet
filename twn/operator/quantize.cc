#include "./quantize-inl.h"
#include <nnvm/op_attr_types.h>

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(QuantizeParam param) {
  Operator* op = NULL;
  op = new QuantizeOp<cpu>(param);
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *QuantizeProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(QuantizeParam);

MXNET_REGISTER_OP_PROPERTY(Quantize, QuantizeProp)
.describe(R"code(Quantize the given input symbol.)code" ADD_FILELINE)
.add_argument("data", "ndarray-or-symbol", "Input data to quantize")
.add_arguments(QuantizeParam::__FIELDS__());

// NNVM_REGISTER_OP(Quantize)
// .set_attr<nnvm::FSetInputVarAttrOnCompose>("FSetInputVarAttrOnCompose",
//     [](const nnvm::NodeAttrs& attrs, nnvm::NodePtr var, const int index) {
//       if (var->attrs.dict.find("__init__") != var->attrs.dict.end()) return;
//       if (index == 1) {
//         var->attrs.dict["__init__"] = "[\"one\", {}]";
//       } 
//     });

}  // namespace op
}  // namespace mxnet


