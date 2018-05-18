<strong>This page is now under construction!</strong>

# LiteFlowNet
This repository is the release of <strong>LiteFlowNet</strong> for our paper <strong>LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation</strong></a> in CVPR18 (Spotlight).

For more details about LiteFlowNet, please refer to <a href="http://mmlab.ie.cuhk.edu.hk/projects/LiteFlowNet/"> my project page</a>.

It comes as a fork of the modified caffe master branch from <a href="https://github.com/lmb-freiburg/flownet2">FlowNet2</a> with new layers, scripts, and trained models.

# Prerequisites
Installation was tested under Ubuntu 14.04.5 and 16.04.2 with CUDA 8.0 and cuDNN 5.1. 

For opencv 3+, you may need to change <code>opencv2/gpu/gpu.hpp</code> to <code>opencv2/cudaarithm.hpp</code> in <code>/LiteFlowNet/src/caffe/layersresample_layer.cu</code>.

If your machine installed a newer version of cuDNN, you do not need to downgrade it. You can do the following trick: 
1. Download <code>cudnn-8.0-linux-x64-v5.1.tgz</code> and untar it to a temp folder, say <code>cuda-8-cudnn-5.1</code>	

2. Rename <code>cudnn.h</code> to <code>cudnn-5.1.h</code> in the folder <code>/cuda-8-cudnn-5.1/include</code>	

3. <pre><code>sudo cp cuda-8-cudnn-5.1/include/cudnn-5.1.h /usr/local/cuda/include/</code></pre>	

4. <pre><code>sudo cp cuda-8-cudnn-5.1/lib64/lib* /usr/local/cuda/lib64/</code></pre>	

5. Replace <code>#include <cudnn.h></code> to <code>#include <cudnn-5.1.h></code> in <code>LiteFlowNet/include/caffe/util/cudnn.hpp</code>.
    
# Compiling
<pre><code>$ cd LiteFlowNet</code></pre>
<pre><code>$ make -j 8 all tools pycaffe</code></pre>

# Datasets
1. <a href="https://lmb.informatik.uni-freiburg.de/data/FlyingChairs/FlyingChairs.zip"> FlyingChairs dataset</a> (31GB) and <a href="https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs/FlyingChairs_train_val.txt">train-validation split</a>.
2. <a href="https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/raw_data/flyingthings3d__frames_cleanpass.tar"> RGB image pairs (clean pass)</a> (37GB) and <a href="https://lmb.informatik.uni-freiburg.de/data/SceneFlowDatasets_CVPR16/Release_april16/data/FlyingThings3D/derived_data/flyingthings3d__optical_flow.tar.bz2"> flow fields</a> (311GB) for Things3D dataset.
3. <a href="http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"> Sintel dataset (clean + final passes)</a> (5.3GB).
4. <a href="http://www.cvlibs.net/download.php?file=data_stereo_flow.zip"> KITTI12 dataset</a> (2GB) and <a href="http://www.cvlibs.net/download.php?file=data_scene_flow.zip"> KITTI15 dataset</a> (2GB) (Simple registration is required).

# Training
1. Prepare the training set. In <code>LiteFlowNet/data/make-lmdbs-train.sh</code>, change <code>YOUR_TRAINING_SET</code> and <code>YOUR_TESTING_SET</code> to your favourite dataset.
<pre><code>$ cd LiteFlowNet/data</code></pre>
<pre><code>$ ./make-lmdbs-train.sh</code></pre>

2. Copy files from <code>TEMPLATE</code> and edit all the prototxt files and make sure all settings are correct
<pre><code>$ cd LiteFlowNet/models/TEMPLATE</code></pre>	
<pre><code>$ cp solver.prototxt.template solver.prototxt</code></pre>	
<pre><code>$ cp train.prototxt.template train.prototxt</code></pre>

3. Create a soft link in your model folder (e.g. <code>LiteFlowNet/models/TEMPLATE</code>)
<pre><code>$ ln -s ../../build/tools bin</code></pre>

4. Run the training script	
<pre><code>$ ./train.py -gpu 0 2>&1 | tee ./log.txt</code></pre>

# Trained models	
The trained models (liteflownet-pre, liteflownet, liteflownet-ft-sintel, liteflownet-ft-kitti) are available in the folder <code>LiteFlowNet/models/trained</code>. Untar the files to the same folder before you use it.

# Testing	
1. Replace "MODEL" to one of the trained models in the line <code>cnn_model = './trained/MODEL'</code> of <code>LiteFlowNet/models/test_MODE.py</code>.

2. Replace MODE to "batch" if all the images has the same resolution (e.g. Sintel), otherwise replace it to "iter" (e.g. KITTI).

3. <pre><code>$ test_MODE.py img1_pathList.txt img2_pathList.txt ./results/YOUR_TESTING_SET</code></pre>

# License and Citation	
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. If our work helps your research or you use the code in your research, please cite the following paper:

<pre><code>@InProceedings{hui18liteflownet,  	
 author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},  	
 title = {LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation},  	
 booktitle  = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  	
 year = {2018},  	
 url = {http://mmlab.ie.cuhk.edu.hk/projects/LiteFlowNet/}	
}</code></pre>
