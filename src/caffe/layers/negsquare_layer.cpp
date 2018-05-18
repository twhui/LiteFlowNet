#include <vector>

#include "caffe/layers/negsquare_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NegSquareLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

// Compute y = -(x)^2r
template <typename Dtype>
void NegSquareLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "NegSquareLayer: CPU Forward not yet implemented.";
}

template <typename Dtype>
void NegSquareLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "NegSquareLayer: CPU Forward not yet implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(NegSquareLayer);
#endif

INSTANTIATE_CLASS(NegSquareLayer);
REGISTER_LAYER_CLASS(NegSquare);

}  // namespace caffe
