#!/usr/bin/env python
import os, sys
import subprocess
from math import ceil

caffe_bin = 'bin/caffe.bin'
img_size_bin = 'bin/get_image_size'

template = './deploy.prototxt'
cnn_model = 'MODEL'   # MODEL = liteflownet, liteflownet-ft-sintel or liteflownet-ft-kitti

# =========================================================
def get_image_size(filename):
    global img_size_bin
    dim_list = [int(dimstr) for dimstr in str(subprocess.check_output([img_size_bin, filename])).split(',')]
    if not len(dim_list) == 2:
        print('Could not determine size of image %s' % filename)
        sys.exit(1)
    return dim_list


def sizes_equal(size1, size2):
    return size1[0] == size2[0] and size1[1] == size2[1]


def check_image_lists(lists):
    images = [[], []]

    with open(lists[0], 'r') as f:
        images[0] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    with open(lists[1], 'r') as f:
        images[1] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    if len(images[0]) != len(images[1]):
        print("Unequal amount of images in the given lists (%d vs. %d)" % (len(images[0]), len(images[1])))
        sys.exit(1)

    if not os.path.isfile(images[0][0]):
        print('Image %s not found' % images[0][0])
        sys.exit(1)

    base_size = get_image_size(images[0][0])

    for idx in range(len(images[0])):
        print("Checking image pair %d of %d" % (idx+1, len(images[0])))
        img1 = images[0][idx]
        img2 = images[1][idx]

        if not os.path.isfile(img1):
            print('Image %s not found' % img1)
            sys.exit(1)

        if not os.path.isfile(img2):
            print('Image %s not found' % img2)
            sys.exit(1)

        img1_size = get_image_size(img1)
        img2_size = get_image_size(img2)

        if not (sizes_equal(base_size, img1_size) and sizes_equal(base_size, img2_size)):
            print('The images do not all have the same size. (Images: %s or %s vs. %s)\n Please use the pair-mode.' % (img1, img2, images[0][idx]))
            sys.exit(1)

    return base_size[0], base_size[1], len(images[0])

my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir)

if not (os.path.isfile(caffe_bin) and os.path.isfile(img_size_bin)):
    print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
    sys.exit(1)

if len(sys.argv)-1 != 3:
    print("Use this tool to test FlowNet on images\n"
          "Usage for single image pair:\n"
          "    ./demo_flownets.py IMAGE1 IMAGE2 OUTPUT_FOLDER\n"
          "\n"
          "Usage for a pair of image lists (must end with .txt):\n"
          "    ./demo_flownets.py LIST1.TXT LIST2.TXT OUTPUT_FOLDER\n")
    sys.exit(1)

img_files = sys.argv[1:]
print("Image files: " + str(img_files))


# Frame-by-frame processing
images = [[], []]

with open(img_files[0], 'r') as f:
    images[0] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
with open(img_files[1], 'r') as f:
    images[1] = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

for idx in reversed(range(len(images[0]))):

    img1_size = get_image_size(images[0][idx])
    img2_size = get_image_size(images[1][idx])

    if not (sizes_equal(img1_size, img2_size)):
        print('The images do not have the same size. (Images: %s or %s vs. %s)\n Please use the pair-mode.' % (img1, img2, images[0][idx]))
        sys.exit(1)

    width = img1_size[0]
    height = img1_size[1]

    # Prepare prototxt
    subprocess.call('mkdir -p tmp', shell=True)

    with open('tmp/img1.txt', "w") as tfile:
       tfile.write("%s\n" % images[0][idx])

    with open('tmp/img2.txt', "w") as tfile:
       tfile.write("%s\n" % images[1][idx])


    divisor = 32.
    adapted_width = ceil(width/divisor) * divisor
    adapted_height = ceil(height/divisor) * divisor
    rescale_coeff_x = width / adapted_width
    rescale_coeff_y = height / adapted_height

    replacement_list = {
        '$ADAPTED_WIDTH': ('%d' % adapted_width),
        '$ADAPTED_HEIGHT': ('%d' % adapted_height),
        '$TARGET_WIDTH': ('%d' % width),
        '$TARGET_HEIGHT': ('%d' % height),
        '$SCALE_WIDTH': ('%.8f' % rescale_coeff_x),
        '$SCALE_HEIGHT': ('%.8f' % rescale_coeff_y),
        '$OUTFOLDER': ('%s' % '"' + img_files[2] + '"'),
        '$CNN': ('%s' % '"' + cnn_model + '-"'),
    }

    proto = ''
    with open(template, "r") as tfile:
        proto = tfile.read()

    for r in replacement_list:
        proto = proto.replace(r, replacement_list[r])

    with open('tmp/deploy.prototxt', "w") as tfile:
        tfile.write(proto)

    # Run caffe
    args = [caffe_bin, 'test', '-model', 'tmp/deploy.prototxt',
            '-weights', '../trained/' + cnn_model + '.caffemodel',
            '-iterations', str(1),
            '-gpu', '0']

    cmd = str.join(' ', args)
    print('Executing %s' % cmd)
    subprocess.call(args)

    if idx > 0:
        os.rename(img_files[2] + '/' + cnn_model + '-0000000.flo', img_files[2] + '/' + cnn_model +'-' + '{0:07d}'.format(idx) + '.flo')


print('\nThe resulting FLOW is stored in CNN-NNNNNNN.flo')
