#ifndef MXNET_OPERATOR_TERNARIZE_INL_H_
#define MXNET_OPERATOR_TERNARIZE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./elemwise_op_common.h"
#include "./mshadow_op_ternary.h"


namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace ternarize {
  enum TernarizeOpInputs {kData};
  enum TernarizeOpAuxiliary {kAuxThres};
  enum TernarizeOpOutputs {kOut, kThres};
}  // fullc

struct TernarizeParam : public dmlc::Parameter<TernarizeParam> {
  float th_zero_ratio;
  bool soft_ternarization;
  DMLC_DECLARE_PARAMETER(TernarizeParam) {
    DMLC_DECLARE_FIELD(th_zero_ratio).set_default(0.7f)
    .describe("to be filled.");
    DMLC_DECLARE_FIELD(soft_ternarization).set_default(false)
    .describe("Apply soft ternarization, just push weights into ternary values.");
  }
};

template<typename xpu, typename DType>
class TernarizeOp : public Operator {
  public:
    explicit TernarizeOp(TernarizeParam p) {
      this->param_ = p;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
      using namespace mshadow;
      using namespace mshadow::expr;
      CHECK_EQ(req[ternarize::kOut], kWriteTo);
      CHECK_EQ(in_data.size(), 1);
      CHECK_EQ(out_data.size(), 2);
      Stream<xpu> *s = ctx.get_stream<xpu>();

      Shape<2> shape_1d = Shape2(1, in_data[ternarize::kData].Size());
      Tensor<xpu, 2, DType> data_1d = 
        in_data[ternarize::kData].get_with_shape<xpu, 2, DType>(shape_1d, s);
      Tensor<xpu, 2, DType> out_1d = 
        out_data[ternarize::kOut].get_with_shape<xpu, 2, DType>(shape_1d, s);
      Tensor<xpu, 1, DType> aux_thres = 
        aux_args[ternarize::kAuxThres].get<xpu, 1, DType>(s); // size 1 tensor
      Tensor<xpu, 1, DType> thres = 
        out_data[ternarize::kThres].get<xpu, 1, DType>(s); // size 1 tensor

      if(ctx.is_train) {
        aux_thres = sumall_except_dim<0>(F<mshadow_op::abs>(data_1d));
        aux_thres /= static_cast<float>(in_data[ternarize::kData].Size());
        aux_thres *= param_.th_zero_ratio;
      }
      if (param_.soft_ternarization) {
        Assign(thres, req[ternarize::kThres], F<mshadow_op::identity>(aux_thres));
        Assign(out_1d, req[ternarize::kOut], 
            F<mshadow_op::ternarize>(data_1d, broadcast<0>(thres, shape_1d)));
      } else {
        thres = DType(1.f);
        Assign(out_1d, req[ternarize::kOut], 
            F<mshadow_op::clip>(data_1d, broadcast<0>(thres, shape_1d)));
      }
      // out_1d *= broadcast<0>(thres, shape_1d);
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
      CHECK_EQ(out_grad.size(), 2);
      CHECK_EQ(in_grad.size(), 1);
      Stream<xpu> *s = ctx.get_stream<xpu>();

      // simply pass the gradient
      Shape<2> shape_1d = Shape2(1, out_grad[ternarize::kOut].Size());
      auto grad = out_grad[ternarize::kOut].get_with_shape<xpu, 2, DType>(shape_1d, s);
      auto gdata = in_grad[ternarize::kData].get_with_shape<xpu, 2, DType>(shape_1d, s);
      auto data_1d = out_data[ternarize::kOut].get_with_shape<xpu, 2, DType>(shape_1d, s);
      auto aux_thres = 
        aux_args[ternarize::kAuxThres].get<xpu, 1, DType>(s); // size 1 tensor

      if (param_.soft_ternarization) {
        Assign(gdata, req[ternarize::kData], F<mshadow_op::identity>(grad));
      } else {
        Assign(gdata, req[ternarize::kData], 
            F<mshadow_op::clip_grad>(data_1d, broadcast<0>(aux_thres, shape_1d)) * grad);
      }
    }

  private:
    TernarizeParam param_;
}; // calss TenarizeOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(TernarizeParam param, int dtype);

class TernarizeProp : public OperatorProperty {
  public:
    void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
      param_.Init(kwargs);
    }

    bool InferShape(std::vector<TShape> *in_shape,
                    std::vector<TShape> *out_shape,
                    std::vector<TShape> *aux_shape) const override {
      using namespace mshadow;
      CHECK_EQ(in_shape->size(), 1) << "Input:[data]";
      // CHECK_EQ(out_shape->size(), 2) << "Output:[out, thres]";
      // copy output data shape to input data shape
      TShape dshape = in_shape->at(0);
      aux_shape->clear();
      aux_shape->push_back(Shape1(1));
      out_shape->clear();
      out_shape->push_back(dshape);
      out_shape->push_back(Shape1(1));
      return true;
    }

    bool InferType(std::vector<int> *in_type,
                   std::vector<int> *out_type,
                   std::vector<int> *aux_type) const override {
      CHECK_GE(in_type->size(), 1);
      nnvm::NodeAttrs attrs;
      attrs.name = "Ternarize";
      bool is_good = ElemwiseAttr<int, type_is_none, type_assign, true>(
        attrs, in_type, out_type, -1);
      aux_type->clear();
      aux_type->push_back(in_type->at(0));
      return is_good;
    }

    std::map<std::string, std::string> GetParams() const override {
      return param_.__DICT__();
    }

    OperatorProperty* Copy() const override {
      auto ptr = new TernarizeProp();
      ptr->param_ = param_;
      return ptr;
    }

    std::string TypeString() const override {
      return "Ternarize";
    }

    std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
      return {out_data[ternarize::kOut], out_grad[ternarize::kOut], };
    }

    int NumOutputs() const override {
      return 2;
    }

    std::vector<std::string> ListArguments() const override {
      return {"data"};
    }

    std::vector<std::string> ListAuxiliaryStates() const override {
      return {"aux_thres"};
    }

    std::vector<std::string> ListOutputs() const override {
      return {"output", "thres"};
    }

    Operator* CreateOperator(Context ctx) const override {
        LOG(FATAL) << "Not Implemented.";
        return NULL;
    }

    Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                               std::vector<int> *in_type) const override;

  private:
    TernarizeParam param_;
}; // class TernarizeProp

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TERNARIZE_INL_H_
