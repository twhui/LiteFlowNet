#include <vector>

#include "caffe/layers/expmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ExpMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  const Dtype base = this->layer_param_.expmax_param().base();
  if (base != Dtype(-1)) {
    CHECK_GT(base, 0) << "base must be strictly positive.";
  }
  // If base == -1, interpret the base as e and set log_base = 1 exactly.
  // Otherwise, calculate its log explicitly.
  const Dtype log_base = (base == Dtype(-1)) ? Dtype(1) : log(base);
  CHECK(!isnan(log_base))
      << "NaN result: log(base) = log(" << base << ") = " << log_base;
  CHECK(!isinf(log_base))
      << "Inf result: log(base) = log(" << base << ") = " << log_base;
  const Dtype input_scale = this->layer_param_.expmax_param().scale();
  const Dtype input_shift = this->layer_param_.expmax_param().shift();
  inner_scale_ = log_base * input_scale;
  outer_scale_ = (input_shift == Dtype(0)) ? Dtype(1) :
     ( (base != Dtype(-1)) ? pow(base, input_shift) : exp(input_shift) );

  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[1] = 1; // softmax_axis_ = 1
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void ExpMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      LOG(FATAL) << "Forward CPU Augmentation not implemented.";
}

template <typename Dtype>
void ExpMaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_mul(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != Dtype(1)) {
    caffe_scal(count, inner_scale_, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(ExpMaxLayer);
#endif

INSTANTIATE_CLASS(ExpMaxLayer);
REGISTER_LAYER_CLASS(ExpMax);

}  // namespace caffe
