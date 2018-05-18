#include <vector>

#include "caffe/layers/negsquare_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NegSquareLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  caffe_copy(count, bottom_data, top_data);

  caffe_gpu_powx(count, top_data, Dtype(2), top_data);
  caffe_gpu_scal(count, Dtype(-1), top_data);
}

template <typename Dtype>
void NegSquareLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();

    // Special case for y = -x^2
    //     -> dy/dx = -2 * x
    caffe_gpu_scal(count, Dtype(-1), bottom_diff);
    caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NegSquareLayer);


}  // namespace caffe
