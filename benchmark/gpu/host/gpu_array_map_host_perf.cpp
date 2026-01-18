#include <bpftime_shm.hpp>

#include <cuda.h>
#include <linux/bpf.h>

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
void bpftime_initialize_global_shm(bpftime::shm_open_type type);
void bpftime_destroy_global_shm();
int bpftime_maps_create(int fd, const char *name, bpftime::bpf_map_attr attr);
long bpftime_map_update_elem(int fd, const void *key, const void *value,
			     uint64_t flags);
const void *bpftime_map_lookup_elem(int fd, const void *key);
}

static uint64_t now_ns()
{
	return static_cast<uint64_t>(
		std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::steady_clock::now().time_since_epoch())
			.count());
}

struct bench_result {
	std::string name;
	uint64_t iters = 0;
	uint64_t total_ns = 0;
	uint64_t checksum = 0;
};

static void print_result(const bench_result &r)
{
	const double ns_per_op =
		r.iters ? static_cast<double>(r.total_ns) /
				  static_cast<double>(r.iters)
			: 0.0;
	const double ops_per_s = ns_per_op ? 1e9 / ns_per_op : 0.0;
	std::printf("%s: iters=%" PRIu64 " total=%" PRIu64
		    " ns  avg=%.1f ns/op  %.2f ops/s  checksum=%" PRIu64 "\n",
		    r.name.c_str(), r.iters, r.total_ns, ns_per_op, ops_per_s,
		    r.checksum);
}

static void require_ok(int rc, std::string_view what)
{
	if (rc < 0) {
		std::fprintf(stderr, "fatal: %.*s failed: rc=%d errno=%d (%s)\n",
			     static_cast<int>(what.size()), what.data(), rc,
			     errno, std::strerror(errno));
		std::exit(2);
	}
}

static int create_gpu_array_map(uint32_t value_size, uint32_t max_entries)
{
	constexpr int kFdAuto = -1;
	bpftime::bpf_map_attr attr = {};
	attr.type = static_cast<int>(
		bpftime::bpf_map_type::BPF_MAP_TYPE_GPU_ARRAY_MAP);
	attr.key_size = sizeof(uint32_t);
	attr.value_size = value_size;
	attr.max_ents = max_entries;
	attr.flags = 0;

	int fd = bpftime_maps_create(kFdAuto, "gpu_array_map_host_perf", attr);
	require_ok(fd, "bpftime_maps_create");
	return fd;
}

static bench_result bench_update(int fd, const std::vector<uint32_t> &keys,
				 uint32_t value_size)
{
	std::vector<uint8_t> value(value_size, 0);
	bench_result r;
	r.name = "update";
	r.iters = keys.size();

	uint64_t start = now_ns();
	for (uint32_t i = 0; i < keys.size(); i++) {
		std::memcpy(value.data(), &i,
			    std::min<size_t>(sizeof(i), value.size()));
		long rc = bpftime_map_update_elem(fd, &keys[i], value.data(),
						 BPF_ANY);
		if (rc != 0) {
			std::fprintf(stderr,
				     "update failed at i=%u key=%u: rc=%ld errno=%d (%s)\n",
				     i, keys[i], rc, errno,
				     std::strerror(errno));
			std::exit(2);
		}
		r.checksum += value[0];
	}
	r.total_ns = now_ns() - start;
	return r;
}

static bench_result bench_lookup(int fd, const std::vector<uint32_t> &keys,
				 uint32_t value_size)
{
	bench_result r;
	r.name = "lookup";
	r.iters = keys.size();

	uint64_t start = now_ns();
	for (uint32_t i = 0; i < keys.size(); i++) {
		const void *ptr = bpftime_map_lookup_elem(fd, &keys[i]);
		if (!ptr) {
			std::fprintf(stderr,
				     "lookup failed at i=%u key=%u: errno=%d (%s)\n",
				     i, keys[i], errno, std::strerror(errno));
			std::exit(2);
		}
		r.checksum += static_cast<const uint8_t *>(ptr)[0];
		if (value_size >= sizeof(uint64_t)) {
			uint64_t v = 0;
			std::memcpy(&v, ptr, sizeof(v));
			r.checksum ^= v;
		}
	}
	r.total_ns = now_ns() - start;
	return r;
}

static std::vector<uint32_t> make_keys(uint32_t iters, uint32_t max_entries)
{
	std::vector<uint32_t> keys;
	keys.reserve(iters);
	// A cheap LCG for key distribution without pulling in <random>.
	uint32_t x = 0x12345678u;
	for (uint32_t i = 0; i < iters; i++) {
		x = x * 1664525u + 1013904223u;
		keys.push_back(x % max_entries);
	}
	return keys;
}

int main(int argc, char **argv)
{
	uint32_t iters = 20000;
	uint32_t max_entries = 1024;
	uint32_t value_size = 8;

	for (int i = 1; i < argc; i++) {
		std::string_view arg(argv[i]);
		auto consume_u32 = [&](std::string_view name, uint32_t &out) {
			if (arg == name && i + 1 < argc) {
				out = static_cast<uint32_t>(
					std::strtoul(argv[++i], nullptr, 10));
				return true;
			}
			return false;
		};
		if (consume_u32("--iters", iters))
			continue;
		if (consume_u32("--max-entries", max_entries))
			continue;
		if (consume_u32("--value-size", value_size))
			continue;
		if (arg == "--help" || arg == "-h") {
			std::printf(
				"Usage: %s [--iters N] [--max-entries N] [--value-size N]\n",
				argv[0]);
			return 0;
		}
		std::fprintf(stderr, "Unknown arg: %.*s\n",
			     static_cast<int>(arg.size()), arg.data());
		return 2;
	}

	std::printf(
		"bpftime GPU_ARRAY_MAP host-side perf (cuMemcpy-based)\n"
		"iters=%u max_entries=%u value_size=%u\n",
		iters, max_entries, value_size);

	if (CUresult err = cuInit(0); err != CUDA_SUCCESS) {
		std::fprintf(stderr, "fatal: cuInit(0) failed: %d\n",
			     static_cast<int>(err));
		return 2;
	}

	bpftime_initialize_global_shm(bpftime::shm_open_type::SHM_REMOVE_AND_CREATE);
	int fd = create_gpu_array_map(value_size, max_entries);

	auto keys = make_keys(iters, max_entries);
	// Warm up (small number of ops) to avoid counting first-call overhead.
	auto warm_keys = make_keys(256, max_entries);
	(void)bench_update(fd, warm_keys, value_size);
	(void)bench_lookup(fd, warm_keys, value_size);

	auto upd = bench_update(fd, keys, value_size);
	auto lkp = bench_lookup(fd, keys, value_size);

	print_result(upd);
	print_result(lkp);

	bpftime_destroy_global_shm();
	return 0;
}
