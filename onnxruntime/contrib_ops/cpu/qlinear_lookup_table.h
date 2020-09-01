// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <vector>

namespace onnxruntime {
namespace contrib {

template <typename T>
class QLinearLookupBase : public OpKernel {
 public:
  QLinearLookupBase(const OpKernelInfo& info)
      : OpKernel(info), fixed_lookup_table_() {
  }

  // function that transform array of input value to array of output value of length
  typedef std::function<void(const float* input, float* input, size_t length)> ArrayValueTransformer;

  // function that transform single value
  typedef std::function<float(float)> ScalarValueTransformer;

 protected:
  template <typename Transformer>
  Status ComputeBase(OpKernelContext* context, Transformer fn) const;

  // Should be called in derived class's constructor
  template <typename Transformer>
  void BuildFixedTableIfPossible(Transformer fn);

  // when input quantizaton parameters are const, pre-compute table value.
  // After construction, non-zero size means pre-computed. Save space when not pre-computed.
  std::vector<uint8_t> fixed_lookup_table_;
};

template <typename T>
class QLinearLeakyRelu final : public QLinearLookupBase {
 public:
  QLinearLeakyRelu(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

 private:
  const float alpha_;
};

template <typename T>
class QLinearSigmoid final : public QLinearLookupBase {
 public:
  QLinearSigmoid(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

}  // namespace contrib
}  // namespace onnxruntime
