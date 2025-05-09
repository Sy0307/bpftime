.PHONY: install benchmark coverage test docs help build clean unit-test-daemon unit-test unit-test-runtime
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z\d_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python3 -c "$$BROWSER_PYSCRIPT"
INSTALL_LOCATION := ~/.local
CXXFLAGS += -std=c++20
JOBS := 1

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

build-unit-test:
	cmake -Bbuild  -DBPFTIME_ENABLE_UNIT_TESTING=1 -DCMAKE_BUILD_TYPE:STRING=Debug -DENABLE_PROBE_WRITE_CHECK=1 -DENABLE_PROBE_READ_CHECK=1
	cmake --build build --config Debug --target bpftime_runtime_tests bpftime_daemon_tests  -j$(JOBS)

build-unit-test-without-probe-check:
	cmake -Bbuild  -DBPFTIME_ENABLE_UNIT_TESTING=1 -DCMAKE_BUILD_TYPE:STRING=Debug
	cmake --build build --config Debug --target bpftime_runtime_tests bpftime_daemon_tests  -j$(JOBS)

unit-test-daemon: 
	./build/daemon/test/bpftime_daemon_tests

unit-test-runtime:
	make -C runtime/test/bpf && cp runtime/test/bpf/*.bpf.o build/runtime/test/
	export BPFTIME_VM_NAME=llvm 
	./build/runtime/unit-test/bpftime_runtime_tests
	cd build/runtime/test && make && ctest -VV

unit-test: unit-test-daemon unit-test-runtime ## run catch2 unit tests

build: ## build the package with test and all components
	cmake -Bbuild -DBPFTIME_ENABLE_UNIT_TESTING=1 -DBUILD_BPFTIME_DAEMON=1 -DCMAKE_BUILD_TYPE:STRING=Debug
	cmake --build build --config Debug  -j$(JOBS)

build-iouring: ## build the package with iouring extension
	cmake -Bbuild -DBPFTIME_ENABLE_IOURING_EXT=1 -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo
	cmake --build build --config RelWithDebInfo  -j$(JOBS)

build-wo-libbpf: ## build the package with iouring extension
	cmake -Bbuild -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -DBPFTIME_BUILD_WITH_LIBBPF=OFF -DBPFTIME_BUILD_KERNEL_BPF=OFF
	cmake --build build --config RelWithDebInfo  --target install -j$(JOBS)

release: ## build the release version
	cmake -Bbuild  -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
				   -DBUILD_BPFTIME_DAEMON=1
	cmake --build build --config RelWithDebInfo --target install  -j$(JOBS)

release-with-llvm-jit: ## build the package, with llvm-jit
	cmake -Bbuild  -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
				   -DBPFTIME_LLVM_JIT=1 \
				   -DBUILD_BPFTIME_DAEMON=1
	cmake --build build --config RelWithDebInfo --target install -j$(JOBS)

release-with-static-lib: ## build the release version with libbpftime archive
	cmake -Bbuild  -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
				   -DBPFTIME_BUILD_STATIC_LIB=ON
	cmake --build build --config RelWithDebInfo --target install  -j$(JOBS)

benchmark: ## build and run the benchmark
	cmake -Bbuild -DLLVM_DIR=/usr/lib/llvm-15/cmake \
		-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
		-DBPFTIME_LLVM_JIT=1 \
		-DBPFTIME_ENABLE_LTO=1 \
		-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO \
		-DENABLE_PROBE_WRITE_CHECK=0 \
		-DENABLE_PROBE_READ_CHECK=0 \
		-DBUILD_ATTACH_IMPL_EXAMPLE=1
	cmake --build build --config RelWithDebInfo --target install -j
	# build the mpk version
	cmake -Bbuild-mpk -DLLVM_DIR=/usr/lib/llvm-15/cmake \
		-DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
		-DBPFTIME_LLVM_JIT=1 \
		-DBPFTIME_ENABLE_LTO=1 \
		-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO \
		-DENABLE_PROBE_WRITE_CHECK=0 \
		-DENABLE_PROBE_READ_CHECK=0 \
		-DBPFTIME_ENABLE_MPK=1
	cmake --build build-mpk --config RelWithDebInfo --target install -j
	cmake --build build --config RelWithDebInfo --target attach_impl_example_nginx -j
	make -C benchmark

run-all-benchmark: ## run all benchmarks
	# run micro-benchmarks
	python3 benchmark/uprobe/benchmark.py
	# run remove to avoid conflict with the previous run
	sudo build/tools/bpftimetool/bpftimetool remove
	python3 benchmark/syscall/benchmark.py
	sudo build/tools/bpftimetool/bpftimetool remove
	python3 benchmark/mpk/benchmark.py

	# run system-benchmarks
	sudo build/tools/bpftimetool/bpftimetool remove
	python3 benchmark/syscount-nginx/benchmark.py
	sudo build/tools/bpftimetool/bpftimetool remove
	python3 benchmark/ssl-nginx/draw_figture.py

build-vm: ## build only the core library
	make -C vm build

build-llvm: ## build with llvm as jit backend
	cmake -Bbuild   -DBPFTIME_ENABLE_UNIT_TESTING=1 \
					-DBPFTIME_LLVM_JIT=1 \
					-DCMAKE_BUILD_TYPE:STRING=Debug
	cmake --build build --config Debug -j$(JOBS)

clean: ## clean the project
	rm -rf build
	make -C runtime clean
	make -C vm clean

install: release ## Invoke cmake to install..

