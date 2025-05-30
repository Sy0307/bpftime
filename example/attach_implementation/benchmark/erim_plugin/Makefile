CC = gcc
CFLAGS = -Wall -O2 -fPIC
LDFLAGS = -shared

# ERIM library path
ERIM_DIR = ./erim/src/erim
ERIM_LIB = ./erim/bin/erim/liberim.a

all: clone-erim check-mpk liberim_filter.so

# Automatically clone the ERIM repository if it doesn't exist
clone-erim:
	@if [ ! -d "./erim" ]; then \
		echo "ERIM repository not found. Cloning from GitHub..."; \
		git clone https://github.com/vahldiek/erim.git ./erim; \
		cd ./erim && git checkout 7232f4762c5ff51035116f9664ec6e7af3b236af; \
		echo "ERIM repository cloned successfully."; \
	else \
		echo "ERIM repository already exists."; \
	fi

# Check if CPU supports MPK (for informational purposes)
check-mpk:
	@echo "Checking if CPU supports Intel MPK..."
	@if grep -q pku /proc/cpuinfo; then \
		echo "MPK support detected"; \
	else \
		echo "WARNING: MPK support not detected in CPU. The library will be built but may fall back to unprotected mode."; \
	fi

# First build the ERIM library
$(ERIM_LIB): clone-erim
	mkdir -p ./erim/bin/erim
	CFLAGS="-fPIC $(CFLAGS)" $(MAKE) -C $(ERIM_DIR)

# Build our filter implementation with ERIM protection
filter_impl.o: filter_impl.c filter_impl.h
	$(CC) $(CFLAGS) -I$(ERIM_DIR) -c filter_impl.c -o filter_impl.o

liberim_filter.so: filter_impl.o $(ERIM_LIB)
	$(CC) $(LDFLAGS) filter_impl.o $(ERIM_LIB) -o liberim_filter.so

# Run a simple test to verify the functionality
test_filter: test_filter.c liberim_filter.so
	# Create a test-specific object file with ERIM functionality
	$(CC) $(CFLAGS) -I. -I$(ERIM_DIR) -c test_filter.c -o test_filter.o
	# Link everything together with proper libraries
	$(CC) -o test_filter test_filter.o -L. -lerim_filter $(ERIM_LIB)
	@echo "Test program built successfully."

test: test_filter
	@echo "================================================================="
	@echo "  Running ERIM protection test"
	@echo "  Note: If your CPU doesn't support Intel MPK, the test will run"
	@echo "        in fallback mode without memory protection."
	@echo "================================================================="
	LD_LIBRARY_PATH=.:$(LD_LIBRARY_PATH) ./test_filter

clean:
	rm -f *.o *.so test_filter
	rm -rf ./erim
.PHONY: all clean test test_filter check-mpk clone-erim 
