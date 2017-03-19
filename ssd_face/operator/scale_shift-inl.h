/*!
 * Copyright (c) 2015 by Contributors
 * \file batch_norm-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_BATCH_NORM_INL_H_
#define MXNET_OPERATOR_BATCH_NORM_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace scaleshift {
enum ScaleShiftInputs {kData, kGamma, kBeta};
enum ScaleShiftOpOutputs {kOut};
}  // namespace batchnorm

struct ScaleShiftParam : public dmlc::Parameter<ScaleShiftParam> {
  bool fix_gamma;
  DMLC_DECLARE_PARAMETER(ScaleShiftParam) {
//     DMLC_DECLARE_FIELD(eps).set_default(1e-3f)
//     .describe("Epsilon to prevent div 0");
//     DMLC_DECLARE_FIELD(momentum).set_default(0.9f)
//     .describe("Momentum for moving average");
    DMLC_DECLARE_FIELD(fix_gamma).set_default(true)
    .describe("Freeze scale (gamma in BN) while training.");
//     DMLC_DECLARE_FIELD(use_global_stats).set_default(false)
//     .describe("Whether use global moving statistics instead of local batch-norm. "
//               "This will force change batch-norm into a scale shift operator.");
//     DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
//     .describe("Output All,normal mean and var");
  }
};

template<typename xpu>
class ScaleShiftOp : public Operator {
 public:
  explicit ScaleShiftOp(ScaleShiftParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(req.size(), 1);
    CHECK_EQ(req[scaleshift::kOut], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const real_t scale = static_cast<real_t>(in_data[scaleshift::kData].shape_[1]) /
                         static_cast<real_t>(in_data[scaleshift::kData].shape_.Size());
    Tensor<xpu, 4> data;
    Tensor<xpu, 4> out;
    if (in_data[scaleshift::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(in_data[scaleshift::kData].shape_[0],
                               in_data[scaleshift::kData].shape_[1], 1, 1);
      data = in_data[scaleshift::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      out = out_data[scaleshift::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[scaleshift::kData].get<xpu, 4, real_t>(s);
      out = out_data[scaleshift::kOut].get<xpu, 4, real_t>(s);
    }
    Tensor<xpu, 1> slope = in_data[scaleshift::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> bias = in_data[scaleshift::kBeta].get<xpu, 1, real_t>(s);

    Assign(out, req[scaleshift::kOut], 
        broadcast<1>(slope, data.shape_) * data + broadcast<1>(bias, data.shape_));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 1);
    CHECK_EQ(in_grad.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data, grad, grad_in;
    // const real_t scale = static_cast<real_t>(out_grad[scaleshift::kOut].shape_[1]) /
    //                      static_cast<real_t>(out_grad[scaleshift::kOut].shape_.Size());
    if (in_data[scaleshift::kData].ndim() == 2) {
      Shape<4> dshape = Shape4(out_grad[scaleshift::kOut].shape_[0],
                               out_grad[scaleshift::kOut].shape_[1], 1, 1);
      data = in_data[scaleshift::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad = out_grad[scaleshift::kOut].get_with_shape<xpu, 4, real_t>(dshape, s);
      grad_in = in_grad[scaleshift::kData].get_with_shape<xpu, 4, real_t>(dshape, s);
    } else {
      data = in_data[scaleshift::kData].get<xpu, 4, real_t>(s);
      grad = out_grad[scaleshift::kOut].get<xpu, 4, real_t>(s);
      grad_in = in_grad[scaleshift::kData].get<xpu, 4, real_t>(s);
    }

    Tensor<xpu, 1> slope = in_data[scaleshift::kGamma].get<xpu, 1, real_t>(s);
    Tensor<xpu, 1> gbias = in_grad[scaleshift::kBeta].get<xpu, 1, real_t>(s);

    // use global statistics with freeze moving mean and var.
    Assign(gbias, req[scaleshift::kBeta], sumall_except_dim<1>(grad));
    Assign(grad_in, req[scaleshift::kData], grad * broadcast<1>(slope, data.shape_));
    if(!param_.fix_gamma) {
      Tensor<xpu, 1> gslope = in_grad[scaleshift::kGamma].get<xpu, 1, real_t>(s);
      Assign(gslope, req[scaleshift::kGamma], sumall_except_dim<1>(grad * data));
    }
  }

 private:
  ScaleShiftParam param_;
};  // class ScaleShiftOp

template<typename xpu>
Operator *CreateOp(ScaleShiftParam param, int dtype);


#if DMLC_USE_CXX11
class ScaleShiftProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[data, gamma, beta]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    in_shape->at(1) = TShape(Shape1(dshape[1]));
    in_shape->at(2) = TShape(Shape1(dshape[1]));
    out_shape->clear();
    out_shape->push_back(dshape);

    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new ScaleShiftProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "ScaleShift";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[scaleshift::kOut],
            in_data[scaleshift::kData],
            in_data[scaleshift::kGamma]
           };
  }

  // std::vector<ResourceRequest> BackwardResource(
  //     const std::vector<TShape> &in_shape) const override {
  //   return {ResourceRequest::kTempSpace};
  // }
  //
  // int NumVisibleOutputs() const override {
  //   return 1;
  // }
  //
  // int NumOutputs() const override {
  //   return 1;
  // }

  std::vector<std::string> ListArguments() const override {
    return {"data", "gamma", "beta"};
  }

  // std::vector<std::string> ListOutputs() const override {
  //   return {"output", "mean", "var"};
  // }
  //
  // std::vector<std::string> ListAuxiliaryStates() const override {
  //   return {"moving_mean", "moving_var"};
  // }

  Operator* CreateOperator(Context ctx) const override {
      LOG(FATAL) << "Not Implemented.";
      return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
      std::vector<int> *in_type) const override;

 private:
  ScaleShiftParam param_;
};  // class ScaleShiftProp

#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_BATCH_NORM_INL_H_

