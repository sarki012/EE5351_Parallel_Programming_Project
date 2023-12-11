########LIBRAIRIES
LIBS_ffmpeg = -lm -lz -lpthread -lavformat -lavcodec -lavutil

LIBS_opencv = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect

LIBS_autres = -lpthread -ldl -lm

LIBS = $(LIBS_autres) $(LIBS_ffmpeg) $(LIBS_opencv)

NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -I/usr/include/opencv2/ -gencode=arch=compute_50,code=\"sm_50,compute_50\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 `pkg-config --libs opencv`
EXE	        = image2
OBJ	        = image2.o image2cu.o

default: $(EXE)

image2cu.o: image2.cu 
	$(NVCC) -c -o $@ image2.cu $(NVCC_FLAGS)

image2.o: image2.cpp
	$(NVCC) -c -o $@ image2.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
