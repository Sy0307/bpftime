# fs-cache

- cache syscall metadata
- block access to certain files

## cache syscall metadata

## Usage

mount fuse:

```sh
# create source and mount point
mkdir -p benchmark/fuse/test_source
echo "hello world" > benchmark/fuse/test_source/test.txt
mkdir -p benchmark/fuse/test_mount
# mount
sudo benchmark/fuse/fs/passthrough benchmark/fuse/test_source benchmark/fuse/test_mount
# sudo python3 benchmark/fuse/fs/passthrough.py benchmark/fuse/test_source benchmark/fuse/test_mount
# it will keep open and print the op count of each syscall, stop it will unmount
```

test baseline:

```console
sudo benchmark/fuse/filebench getdents64 benchmark/fuse/test_mount
```

run server:

```console
sudo BPFTIME_SHM_MEMORY_MB=1024 LD_PRELOAD=build/runtime/syscall-server/libbpftime-syscall-server.so benchmark/fuse/bpf/fs-cache
```

run agent

```console
# sudo AGENT_SO=build/runtime/agent/libbpftime-agent.so LD_PRELOAD=build/attach/text_segment_transformer/libbpftime-agent-transformer.so  benchmark/fuse/filebench getdents64 benchmark/fuse/test_mount
```

## Old benchmark result (2023)

### cache for getdents64

base line of getdents64 syscall

- Tested with Passthrough fuse:

start fuse:

```console
source ~/OpenCopilot/venv/bin/activate 
python3 passthrough.py  ~/linux ./data/
```

```console
benchmark/fuse/filebench getdents64 benchmark/fuse/test_mount
```

use cache for getdents64 syscall

```console
# AGENT_SO=build/runtime/agent/libbpftime-agent.so LD_PRELOAD=build/runtime/agent-transformer/libbpftime-agent-transformer.so  /home/yunwei/bpftime/example/fs-filter-cache/bench getdents64  /home/yunwei/bpftime-evaluation/fuse/data/virt
....
inode=4294967295 offset=32 reclen=24 type=0 name=.
inode=4294967295 offset=64 reclen=24 type=0 name=..
inode=4294967295 offset=96 reclen=24 type=0 name=kvm
inode=4294967295 offset=128 reclen=24 type=0 name=lib
inode=4294967295 offset=160 reclen=32 type=0 name=Makefile
Average time usage 1761.278320 ns, iter 100000 times

[2023-11-23 23:38:52][info][414216] Compiling using LLJIT
INFO [414216]: Global shm destructed
```

Test with loggerFS:

```sh
sudo loggedfs -l ./log.txt /home/yunwei/bpftime/daemon
```

No cache:

```console
root@yunwei37server:/home/yunwei/bpftime# /home/yunwei/bpftime/example/fs-filter-cache/bench getdents64 /home/yunwei/bpftime/daemon
inode=1710896 offset=40 reclen=32 type=8 name=README.md
inode=1761315 offset=72 reclen=24 type=4 name=user
inode=1753980 offset=104 reclen=24 type=4 name=.
inode=1710893 offset=144 reclen=32 type=8 name=.gitignore
inode=1717723 offset=176 reclen=24 type=4 name=..
inode=1754009 offset=208 reclen=32 type=4 name=kernel
inode=1761304 offset=240 reclen=24 type=4 name=test
inode=1710897 offset=288 reclen=40 type=8 name=bpf_tracer_event.h
inode=1710894 offset=328 reclen=40 type=8 name=CMakeLists.txt
Average time usage 74006.800000 ns, iter 100 times
```

cache:

```console
root@yunwei37server:/home/yunwei/bpftime# AGENT_SO=build/runtime/agent/libbpftime-agent.so LD_PRELOAD=build/runtime/agent-transformer/libbpftime-agent-transformer.so  /home/yunwei/bpftime/example/fs-filter-cache/bench getdents64 /home/yunwei/bpftime/daemon
...
inode=1710896 offset=40 reclen=32 type=8 name=README.md
inode=1761315 offset=72 reclen=24 type=4 name=user
inode=1753980 offset=104 reclen=24 type=4 name=.
inode=1710893 offset=144 reclen=32 type=8 name=.gitignore
inode=1717723 offset=176 reclen=24 type=4 name=..
inode=1754009 offset=208 reclen=32 type=4 name=kernel
inode=1761304 offset=240 reclen=24 type=4 name=test
inode=1710897 offset=288 reclen=40 type=8 name=bpf_tracer_event.h
inode=1710894 offset=328 reclen=40 type=8 name=CMakeLists.txt
Average time usage 1837.819106 ns, iter 1000000 times
```

### filter for openat

filter certain path in control plane, restrict access to certain files with uid based filter

```c
    struct open_args_t args = {
        .fname = "fuse/data/arch/",
        .flags = 0,
        .fname_len = strlen("fuse/data/arch/"),
    };
    unsigned int uid = 0;
    bpf_map_update_elem(bpf_map__fd(obj->maps.open_file_filter), 
    &uid, &args, BPF_ANY);
```

and also filter from fuse:

```py
        if full_path.startswith("/home/yunwei/linux/arch"):
            # throw error
            raise OSError(errno.EACCES, "")
```

Test when fuse rejects openat syscall

```console
root@yunwei37server:/home/yunwei/bpftime-evaluation# /home/yunwei/bpftime-evaluation/fuse/fs-filter-cache/bench open fuse/data/arch/alpha
Average time usage 169593.214670 ns, iter 100000 times
```

Test if we filter early in bpftime

```console
root@yunwei37server:/home/yunwei/bpftime-evaluation# AGENT_SO=../bpftime/build/runtime/agent/libbpftime-agent.so LD_PRELOAD=../bpftime/build/runtime/agent-transformer/libbpftime-agent-transformer.so /home/yunwei/bpftime-evaluation/fuse/fs-filter-cache/bench open fuse/data/arch/alpha
[2023-11-24 02:03:25.486] [info] [agent-transformer.cpp:33] Entering bpftime syscal transformer agent
[2023-11-24 02:03:25.486] [info] [agent-transformer.cpp:61] Using agent ../bpftime/build/runtime/agent/libbpftime-agent.so
[2023-11-24 02:03:25.486] [info] [text_segment_transformer.cpp:239] Page zero setted up..
[2023-11-24 02:03:25.486] [info] [text_segment_transformer.cpp:267] Rewriting executable segments..
[2023-11-24 02:03:26.046] [info] [bpftime_shm_internal.cpp:597] Global shm constructed. shm_open_type 1 for bpftime_maps_shm
[2023-11-24 02:03:26.046] [info] [agent.cpp:81] Initializing agent..
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_enter_openat
[2023-11-24 02:03:26][info][439292] Registered syscall enter hook for openat with perf fd 0
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_enter_open
[2023-11-24 02:03:26][info][439292] Registered syscall enter hook for open with perf fd 19
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_exit_open
[2023-11-24 02:03:26][info][439292] Registered syscall exit hook for open with perf fd 20
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_exit_openat
[2023-11-24 02:03:26][info][439292] Registered syscall exit hook for openat with perf fd 22
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_enter_close
[2023-11-24 02:03:26][info][439292] Registered syscall enter hook for close with perf fd 23
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_enter_getdents64
[2023-11-24 02:03:26][info][439292] Registered syscall enter hook for getdents64 with perf fd 24
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_exit_getdents64
[2023-11-24 02:03:26][info][439292] Registered syscall exit hook for getdents64 with perf fd 25
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_enter_newfstatat
[2023-11-24 02:03:26][info][439292] Registered syscall enter hook for newfstatat with perf fd 26
[2023-11-24 02:03:26][info][439292] Creating tracepoint for tp name sys_enter_statfs
[2023-11-24 02:03:26][info][439292] Registered syscall enter hook for statfs with perf fd 27
[2023-11-24 02:03:26][info][439292] Attach successfully
[2023-11-24 02:03:26][info][439292] Agent syscall trace setup exiting..
[2023-11-24 02:03:26.053] [info] [agent-transformer.cpp:82] Transformer exiting, trace will be usable now
Average time usage 740.980190 ns, iter 100000 times

INFO [439292]: Global shm destructed
```

### filebench with LoggerFS

Test with loggerFS:

```sh
./lologgedfs -l ./log.txt /root/bpftime-evaluation/fuse/tmp
```

Test with filebench:

```sh
filebench -f bench/filemicro_statfile.f
```

Baseline:

```console
root@banana:~/bpftime-evaluation/fuse# filebench -f bench/filemicro_statfile.f
Filebench Version 1.5-alpha3
0.000: Allocated 177MB of shared memory
0.002: Stat File Version 1.0 personality successfully loaded
0.002: Populating and pre-allocating filesets
0.006: bigfileset populated: 10000 files, avg. dir. width = 20, avg. dir. depth = 3.1, 0 leafdirs, 1250.000MB total size
0.006: Removing bigfileset tree (if exists)
0.010: Pre-allocating directories in bigfileset tree
0.021: Pre-allocating files in bigfileset tree
0.987: Waiting for pre-allocation to finish (in case of a parallel pre-allocation)
0.987: Population and pre-allocation of filesets completed
0.987: Starting 1 examinefiles instances
1.994: Running...
62.000: Run took 60 seconds...
62.000: Per-Operation Breakdown
statfile1            18168487ops   302781ops/s   0.0mb/s    0.002ms/op [0.001ms - 0.472ms]
62.000: IO Summary: 18168487 ops 302781.028 ops/s 0/0 rd/wr   0.0mb/s 0.002ms/op
62.000: Shutting down processes
```
