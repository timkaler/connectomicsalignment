
#CC := /home/amatveev/Pipeline/cilkplus-install/bin/g++
#LD := /home/amatveev/Pipeline/cilkplus-install/bin/g++

CC := g++
LD := g++
#CC := /home/armafire/tools/cilkplus-install/bin/g++
#LD := /home/armafire/tools/cilkplus-install/bin/g++

#CC := /home/ubuntu/Parallel-IR/build/bin/clang++
#LD := /home/ubuntu/Parallel-IR/build/bin/clang++
#CC := clang
#LD := clang
#LD := clang++

#OPENCV_INCLUDE := /home/amatveev/Pipeline/tools/OpenCV/opencv-2.4-install/include/
#OPENCV_LIB := /home/amatveev/Pipeline/tools/OpenCV/opencv-2.4-install/lib/

OPENCV_INCLUDE=/home/armafire/tools/opencv-3-install-test/include
OPENCV_LIB=/home/armafire/tools/opencv-3-install-test/lib

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

#CFLAGS += --param inline-unit-growth=1000
#CFLAGS += -mrtm

ifdef DEBUG
	CFLAGS += -O0 -g3
else
	CFLAGS += -DNDEBUG
	CFLAGS += -O3 -g 
	#-Ofast
endif

LDFLAGS += -L$(OPENCV_LIB) -lcilkrts -lopencv_imgcodecs -lopencv_core -lopencv_imgproc -lopencv_features2d -lopencv_xfeatures2d -lopencv_flann -lopencv_video -lopencv_calib3d -lopencv_hdf -lhdf5_hl -lhdf5

BINS = run_align

.PHONY:	all clean

all: $(BINS)

img_io.o: ezsift/img_io.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

ezsift.o: ezsift/ezsift.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

common.o: common.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

align.o: align.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<

run.o: run.cpp
	$(CC) $(CFLAGS) $(DEFINES) -c -o $@ $<    	

run_align: common.o align.o run.o ezsift.o img_io.o
	$(LD) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(BINS) *.o
