# Compiler and flags
CC = nvcc
CFLAGS = --default-stream per-thread  # Flag generici per CUDA
COMPUTE_ARCH = 
SM_ARCH = 
ARCH_FLAGS = -arch=$(COMPUTE_ARCH) -code=$(SM_ARCH)

# Paths (Windows-style, ma con / invece di \)
INCLUDE_PATH = -I""
LIB_PATH = -L""
LIBS = -lmpir #or gmp

# Files
EXEC = main
SRC = PollardRhoGpu.cu
HEADER = PollardRhoGpu.h
OBJ = $(SRC:.cu=.o)

# Rules
all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(ARCH_FLAGS) $(INCLUDE_PATH) $(LIB_PATH) -o $@ $^ $(LIBS)

%.o: %.cu $(HEADER)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) $(INCLUDE_PATH) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)

rebuild: clean all

.PHONY: all clean rebuild
