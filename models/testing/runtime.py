#!/usr/bin/env python
import os, sys
import subprocess

caffe_bin = 'bin/caffe.bin'

# =========================================================
my_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(my_dir)

if not os.path.isfile(caffe_bin):
    print('Caffe tool binaries not found. Did you compile caffe with tools (make all tools)?')
    sys.exit(1)

print('args:', sys.argv[1:])

args = [caffe_bin, 'time', '-model', './model/testing/deploy_runtime_1024_448.prototxt', '-weights', './models/trained/liteflownet-ft-sintel.caffemodel', '-gpu', '0', '-iterations', '100'] + sys.argv[1:]
cmd = str.join(' ', args)
print('Executing %s' % cmd)

subprocess.call(args)
