This page is now under construction!

# LiteFlowNet
This repository is the release of <strong>LiteFlowNet</strong> for our paper <strong>LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation</strong></a> in CVPR18 (Spotlight).

For more details about LiteFlowNet, please refer to <a href="http://mmlab.ie.cuhk.edu.hk/projects/LiteFlowNet/"> my project page</a>.

It comes as a fork of the modified caffe master branch from <a href="https://github.com/lmb-freiburg/flownet2">FlowNet2</a>, new layers, scripts, and trained models.

# Prerequisites
Installation was tested under Ubuntu 14.04.5 and 16.04.2 with CUDA 8.0 and cuDNN 5.1. 

For opencv 3+, you may need to change <code>opencv2/gpu/gpu.hpp</code> to <code>opencv2/cudaarithm.hpp</code> in <code>/liteflownet/src/caffe/layersresample_layer.cu</code>.

If your machine installed a newer version of cuDNN, you do not need to downgrade it. You can do the following trick: 
1. Download <code>cudnn-8.0-linux-x64-v5.1.tgz</code> and untar it to a temp folder, say <code>cuda-8-cudnn-5.1</code>
2. Rename <code>cudnn.h</code> to <code>cudnn-5.1.h</code> in the folder <code>/cuda-8-cudnn-5.1/include</code>
3. <pre><code>sudo cp cuda-8-cudnn-5.1/include/cudnn-5.1.h /usr/local/cuda/include/</code></pre>
5. <pre><code>sudo cp cuda-8-cudnn-5.1/lib64/lib* /usr/local/cuda/lib64/</code></pre>
6. Replace <code>#include <cudnn.h></code> to <code>#include <cudnn-5.1.h></code> in <code>liteflownet/include/caffe/util/cudnn.hpp</code>. 

# Compiling
<pre><code>$ make -j 8 all tools pycaffe</code></pre>

# Trained models, training and testing codes
(To appear)

# License and Citation
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. If our work helps your research or you use the code in your research, please cite the following paper:

<pre><code>@InProceedings{hui18liteflownet,  
  author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},  
  title  = {LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation},  
  booktitle  = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
  year = {2018},  
  url = {http://mmlab.ie.cuhk.edu.hk/projects/LiteFlowNet/}
}</code></pre>
