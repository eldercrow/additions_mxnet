#ifndef MXNET_OPERATOR_QUANTIZE_INL_H_
#define MXNET_OPERATOR_QUANTIZE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "./mshadow_op_quantize.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace quantize {
  enum QuantizeOpInputs {kData};
  enum QuantizeOpOutputs {kOut, kGamma, kBeta};
  enum QuantizeOpResource {kTempSpace, kRandom};
}  // quantize

struct QuantizeParam : public dmlc::Parameter<QuantizeParam> {
  int quantize_bit;
  float th_min;
  float th_max;
  DMLC_DECLARE_PARAMETER(QuantizeParam) {
    DMLC_DECLARE_FIELD(quantize_bit).set_default(8)
      .describe("Quantization bit. Default is 8-bit quantization.");
    DMLC_DECLARE_FIELD(th_min).set_default(0.f)
    .describe("Minimum value to be quantized, smaller than th_min will be clipped");
    DMLC_DECLARE_FIELD(th_max).set_default(5.f)
    .describe("Maximum value to be quantized, bigger than th_max will be clipped");
  }
};

template<typename xpu>
class QuantizeOp : public Operator {
  public:
    typedef float DType;
  public:
    explicit QuantizeOp(QuantizeParam p) {
      this->param_ = p;
      quantize_level_ = powf(2.f, static_cast<float>(param_.quantize_bit)) - 1.f;
      scale_ = quantize_level_ / (param_.th_max - param_.th_min);
      noise_scale_ = (param_.th_max - param_.th_min) * 
                     0.5f / powf(2.f, static_cast<float>(param_.quantize_bit));
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(req[quantize::kOut], kWriteTo);
      CHECK_EQ(in_data.size(), 1);
      CHECK_EQ(out_data.size(), 3);
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Shape<2> shape_1d = Shape2(1, in_data[quantize::kData].Size());
      Tensor<xpu, 2, DType> data_1d = 
        in_data[quantize::kData].get_with_shape<xpu, 2, DType>(shape_1d, s);
      Tensor<xpu, 2, DType> out_1d = 
        out_data[quantize::kOut].get_with_shape<xpu, 2, DType>(shape_1d, s);
      Tensor<xpu, 1, DType> gamma = 
        out_data[quantize::kGamma].get<xpu, 1, DType>(s); // size 1 tensor
      Tensor<xpu, 1, DType> beta = 
        out_data[quantize::kBeta].get<xpu, 1, DType>(s); // size 1 tensor

      Assign(out_1d, req[quantize::kOut], 
              F<mshadow_op::clamp_zero_max>(
                (data_1d - param_.th_min) * scale_, 
                quantize_level_)
          );
      gamma = 1.f / scale_;
      beta = param_.th_min;
    }

    virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr; 
      CHECK_EQ(out_grad.size(), 3);
      CHECK_EQ(in_grad.size(), 1);
      Stream<xpu> *s = ctx.get_stream<xpu>();
      Shape<2> data_shape_1d = Shape2(1, out_grad[quantize::kOut].Size());
      Tensor<xpu, 2, DType> m_out_grad = 
          out_grad[quantize::kOut].get_with_shape<xpu, 2, DType>(data_shape_1d, s);
      Tensor<xpu, 2, DType> m_out_data = 
          out_data[quantize::kOut].get_with_shape<xpu, 2, DType>(data_shape_1d, s);
      Tensor<xpu, 2, DType> m_in_grad = 
          in_grad[quantize::kData].get_with_shape<xpu, 2, DType>(data_shape_1d, s);
      // temp space for noise
      Random<xpu>* prnd = ctx.requested[quantize::kRandom].get_random<xpu, DType>(s);
      Shape<3> temp_shape = Shape3(3, 1, out_grad[quantize::kOut].Size());
      Tensor<xpu, 3, DType> gnoise = 
        ctx.requested[quantize::kTempSpace].get_space_typed<xpu, 3, DType>(temp_shape, s);
      gnoise[0] = tcast<DType>(prnd->uniform(data_shape_1d)) * 2.f - 1.f;
      gnoise[0] *= noise_scale_;

      gnoise[1] = DType(0);
      gnoise[2] = DType(quantize_level_);
      Assign(m_in_grad, req[quantize::kData], 
          F<mshadow_op::clamp_grad>(m_out_data, gnoise[1], gnoise[2]) *
          (m_out_grad*quantize_level_ + gnoise[0]));
    }

  private:
    QuantizeParam param_;
    float quantize_level_;
    float scale_;
    float noise_scale_;
}; // calss QuantizeOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(QuantizeParam param);

class QuantizeProp : public OperatorProperty {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
      param_.Init(kwargs);
    }

    bool InferShape(std::vector<TShape> *in_shape,
                    std::vector<TShape> *out_shape,
                    std::vector<TShape> *aux_shape) const override {
      using namespace mshadow;
      CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
      // CHECK_EQ(out_shape->size(), 3) << "Output:[out, gamma, beta]";
      // copy output data shape to input data shape
      TShape dshape = in_shape->at(0);
      out_shape->clear();
      out_shape->push_back(dshape);
      out_shape->push_back(Shape1(1));
      out_shape->push_back(Shape1(1));
      return true;
    }

    bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      CHECK_GE(in_type->size(), 1);
      nnvm::NodeAttrs attrs;
      attrs.name = "Quantize";
      return ElemwiseAttr<int, type_is_none, type_assign, true, type_string>(
        attrs, in_type, out_type, -1);
    }

    std::map<std::string, std::string> GetParams() const override {
      return param_.__DICT__();
    }

    OperatorProperty* Copy() const override {
      auto ptr = new QuantizeProp();
      ptr->param_ = param_;
      return ptr;
    }

    std::string TypeString() const override {
      return "Quantize";
    }

    std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
      return {out_grad[quantize::kOut], out_data[quantize::kOut]};
    }

    std::vector<ResourceRequest> BackwardResource(
        const std::vector<TShape> &in_shape) const override {
      return {ResourceRequest::kTempSpace, ResourceRequest::kRandom};
    }

    // std::vector<std::pair<int, void*> > BackwardInplaceOption(
    //   const std::vector<int> &out_grad,
    //   const std::vector<int> &in_data,
    //   const std::vector<int> &out_data,
    //   const std::vector<void*> &in_grad) const override {
    //   return {{out_grad[quantize::kOut], in_grad[quantize::kData]}};
    // }
    //
    // std::vector<std::pair<int, void*> > ForwardInplaceOption(
    //   const std::vector<int> &in_data,
    //   const std::vector<void*> &out_data) const override {
    //   return {{in_data[quantize::kData], out_data[quantize::kOut]}};
    // }

    int NumOutputs() const override {
      return 3;
    }

    std::vector<std::string> ListArguments() const override {
      return {"data"};
    }

    std::vector<std::string> ListOutputs() const override {
      return {"output", "gamma", "beta"};
    }

    Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
    }

    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                               std::vector<int> *in_type) const override;

  private:
    QuantizeParam param_;
}; // class QuantizeProp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_QUANTIZE_INL_H_

