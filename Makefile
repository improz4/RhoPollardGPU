# Compiler and flags
CC = nvcc
COMPUTE_ARCH = compute_75
SM_ARCH = sm_75
ARCH_FLAGS = -arch=$(COMPUTE_ARCH) -code=$(SM_ARCH)

# Paths (Windows-style, ma con / invece di \)
INCLUDE_PATH = -I"C:/Users/FrancescoStudente/vcpkg/installed/x64-windows/include"
LIB_PATH = -L"C:/Users/FrancescoStudente/vcpkg/installed/x64-windows/lib"
LIBS = -lmpir -lcurand

# Common flags
CFLAGS = $(ARCH_FLAGS) $(INCLUDE_PATH) $(LIB_PATH) $(LIBS) -O3

# Files
MAIN_EXEC = main.exe
SRC = PollardRhoGpu.cu
HEADER = PollardRhoGpu.h
MAIN_SRC = main.cu
MAIN_OBJ = main.obj
GPU_OBJ = PollardRhoGpu.obj

# Targets
all: $(MAIN_EXEC)

$(MAIN_EXEC): $(MAIN_OBJ) $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $@ $^

$(MAIN_OBJ): $(MAIN_SRC) $(HEADER)
	$(CC) $(CFLAGS) -c -o $@ $<

$(GPU_OBJ): $(SRC) $(HEADER)
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(MAIN_EXEC) $(MAIN_OBJ) $(TEST_OBJ) $(GPU_OBJ) *.exp *.lib

rebuild: clean all
	
.PHONY: all clean rebuild