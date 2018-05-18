#!/bin/bash

../build/tools/convert_imageset_and_flow.bin YOUR_TRAINING_SET.list YOUR_TRAINING_SET_lmdb 0 lmdb
../build/tools/convert_imageset_and_flow.bin YOUR_TESTING_SET.list YOUR_TESTING_SET_lmdb 0 lmdb
