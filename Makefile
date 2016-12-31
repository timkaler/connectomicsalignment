
CC := ./setup.sh $(CXX) #g++
LD := ./setup.sh $(CXX) #g++

OPENCV_LIB=/efs/tools/OpenCV3/lib/
OPENCV_INCLUDE=/efs/tools/OpenCV3/include/
#OPENCV_INCLUDE=/home/armafire/tools/opencv-3-install-test/include/
#OPENCV_LIB=/home/armafire/tools/opencv-3-install-test/lib/

#=======
#OPENCV_ROOT=/efs/home/lemon510/cv
#OPENCV_INCLUDE=$(OPENCV_ROOT)/include
#OPENCV_LIB=$(OPENCV_ROOT)/lib
#>>>>>>> 84dd39cb1a6e33998c46f5e549f732adc98b358f

#CFLAGS += -fcilkplus -m64 -ffast-math -mfma 
CFLAGS += -std=c++11 -ftapir -m64 -march=native -fno-exceptions
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

ifdef LOGIMAGES
  CFLAGS += -DLOGIMAGES
endif

ifdef SKIPHDF5
  CFLAGS += -DSKIPHDF5
endif

ifdef SKIPJSON
  CFLAGS += -DSKIPJSON
endif

ifdef ASSEMBLY
	CFLAGS += -S
endif

ifdef DEBUG
	CFLAGS += -O0 -g3 
else
	CFLAGS += -DNDEBUG
	CFLAGS += -O3 -g -mavx2 -m64 -march=native
	#-Ofast
endif

LDFLAGS += -L$(OPENCV_LIB) -lcilkrts -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d -lopencv_video -lopencv_calib3d -lopencv_hdf -lhdf5_hl -lhdf5 -lgomp


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

match.o: match.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

align.o: align.cpp othersift.cpp gaussianPyramid.cpp boxFilter.cpp filterengine.cpp rowsum.cpp columnsum.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

align.S: align.cpp othersift.cpp gaussianPyramid.cpp boxFilter.cpp filterengine.cpp rowsum.cpp columnsum.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

run.o: run.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<    	

run.S: run.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<    	

run_align: common.o align.o run.o ezsift.o img_io.o match.o
	$(LD) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) *.o
