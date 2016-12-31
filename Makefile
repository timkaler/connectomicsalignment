
#CC := /home/amatveev/Pipeline/cilkplus-install/bin/g++
#LD := /home/amatveev/Pipeline/cilkplus-install/bin/g++

#CC := ./setup.sh $(SETUP) g++
CC := ./setup.sh g++
LD := ./setup.sh g++
#CC := /home/armafire/tools/cilkplus-install/bin/g++
#LD := /home/armafire/tools/cilkplus-install/bin/g++

#CC := /home/ubuntu/Parallel-IR/build/bin/clang++
#LD := /home/ubuntu/Parallel-IR/build/bin/clang++
#CC := clang
#LD := clang
#LD := clang++

#OPENCV_INCLUDE := /home/amatveev/Pipeline/tools/OpenCV/opencv-2.4-install/include/
#OPENCV_LIB := /home/amatveev/Pipeline/tools/OpenCV/opencv-2.4-install/lib/

OPENCV_ROOT=/efs/home/lemon510/cv
OPENCV_INCLUDE=$(OPENCV_ROOT)/include
OPENCV_LIB=$(OPENCV_ROOT)/lib

#CFLAGS += -fcilkplus -m64 -ffast-math -mfma 
CFLAGS += -std=c++11 -fcilkplus -m64 -march=native 
#CFLAGS += -std=c++11 -fdetach
#CFLAGS += -fdetach
#CFLAGS += -fcilkplus -m64 -march=native 
#-ffast-math -mfma -funroll-loops -flto
#CFLAGS += -D_REENTRANT
CFLAGS += -Wall 
#-Winline
CFLAGS += -I$(OPENCV_INCLUDE)


ifdef PROFILE
  CFLAGS += -DPROFILE
  LDFLAGS += -lprofiler
endif

ifdef SKIPOUTPUT
  CFLAGS += -DSKIPOUTPUT
endif

#CFLAGS += --param inline-unit-growth=1000
#CFLAGS += -mrtm

ifdef ASSEMBLY
	CFLAGS += -S
endif

ifdef DEBUG
	CFLAGS += -O0 -g3 
else
	CFLAGS += -DNDEBUG
	CFLAGS += -O3 -g -m64 -march=native -mavx2
endif

LDFLAGS += -L$(OPENCV_LIB) -lcilkrts -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d -lopencv_flann -lopencv_video -lopencv_calib3d -lopencv_hdf -lhdf5_hl -lhdf5 -lgomp

#ifdef DEBUG
#	LDFLAGS += -pg
#endif

ifdef ASSEMBLY
	BINS = img_io.S ezsift.S common.S align.S run.S
else
	BINS = run_align
endif

.PHONY:	all clean

all: $(BINS)

img_io.o: ezsift/img_io.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

img_io.S: ezsift/img_io.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

ezsift.o: ezsift/ezsift.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

ezsift.S: ezsift/ezsift.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

common.o: common.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

common.S: common.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

align.o: align.cpp othersift.cpp gaussianPyramid.cpp boxFilter.cpp filterengine.cpp rowsum.cpp columnsum.cpp minmaxfilter.cpp tests.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

align.S: align.cpp othersift.cpp gaussianPyramid.cpp boxFilter.cpp filterengine.cpp rowsum.cpp columnsum.cpp minmaxfilter.cpp tests.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

run.o: run.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<    	

run.S: run.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<    	

run_align: common.o align.o run.o ezsift.o img_io.o
	$(LD) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) *.o
