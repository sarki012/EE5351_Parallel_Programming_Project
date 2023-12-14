########LIBRAIRIES
LIBS_ffmpeg = -lm -lz -lpthread -lavformat -lavcodec -lavutil

LIBS_opencv = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lopencv_objdetect

LIBS_autres = -lpthread -ldl -lm

LIBS = $(LIBS_autres) $(LIBS_ffmpeg) $(LIBS_opencv)

NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -I/usr/include/opencv4/ -gencode=arch=compute_50,code=\"sm_50,compute_50\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64 `pkg-config --libs opencv4`

EXE	        = histogram_private
#LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64  `-I/usr/local/include/opencv -I/usr/local/#include -L/usr/local/lib -lopencv_shape -lopencv_stitching -lopencv_objdetect -lopencv_superres #-lopencv_videostab -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -#lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -#lopencv_viz -lopencv_core`

OBJ	        = histogram_private.o histogram_private_kernel.o

default: $(EXE)

histogram_private_kernel.o: histogram_private_kernel.cu 
	$(NVCC) -c -o $@ histogram_private_kernel.cu $(NVCC_FLAGS)

histogram_private.o: histogram_private.cpp
	$(NVCC) -c -o $@ histogram_private.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
