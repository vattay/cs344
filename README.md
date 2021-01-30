cs344
=====

Introduction to Parallel Programming class code

# Building on Linux
You will need a 2.x version of OpenCV installed. It's just used to verify your homework. This is getting annoying to build on modern systems, I had to disable CUDA support as OpenCV2 doesn't cmake successfully due to CUDA API divergence. Using OpenCV 2.6.14 I was able to use this to disable CUDA during compilation.

```
mkdir opencv_build
cd opencv_build
cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF ../opencv
cd ../opencv
make
```

# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit. 
[This entry in the Udacity discussion forums](http://forums.udacity.com/questions/100132476/cuda-55-opencv-247-os-x-maverick-it-doesnt-work) describes exactly how to build a compatible OpenCV.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

