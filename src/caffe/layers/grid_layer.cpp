#include <vector>
#include <iostream>
#include "caffe/layers/grid_layer.hpp"

namespace caffe {

template<typename Dtype>
void GridLayer< Dtype >::LayerSetUp(const vector< Blob< Dtype >* >& bottom,
                                    const vector< Blob< Dtype >* >& top) 
{
	// Nothing to setup for
}

template<typename Dtype>
void GridLayer< Dtype >::Reshape(const vector< Blob< Dtype >* >& bottom,
                                 const vector< Blob< Dtype >* >& top) 
{
  this->layer_param_.set_reshape_every_iter(false);
  LOG(WARNING) << "GridLayer only runs Reshape on setup";
 
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  
  CHECK_GE(height_, 1) << "GridLayer must have top_height > 0";
  CHECK_GE(width_, 1) << "GridLayer must have top_width > 0";
  
  top[0]->Reshape(num_, 2, height_, width_);

}

template <typename Dtype>
void GridLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    				   const vector<Blob<Dtype>*>& top) 
{
	Dtype* top_data = top[0]->mutable_cpu_data();

	for ( int n = 0; n < num_; n++ ) {
		for ( int h = 0; h < height_; h++ ) {
			for ( int w = 0; w < width_; w++ ) {
				top_data[((n*2)*height_+h)*width_+w] = w;
				top_data[((n*2+1)*height_+h)*width_+w] = h;
			}
		}
	}	
}

template <typename Dtype>
void GridLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    				    const vector<bool>& propagate_down,
    				    const vector<Blob<Dtype>*>& bottom) 
{

  for(int i=0; i<propagate_down.size(); i++)
        if(propagate_down[i])
                LOG(FATAL) << "GridLayer cannot do backward.";
}

#ifdef CPU_ONLY
STUB_GPU(GridLayer);
#endif

INSTANTIATE_CLASS(GridLayer);
REGISTER_LAYER_CLASS(Grid);

}  // namespace caffe