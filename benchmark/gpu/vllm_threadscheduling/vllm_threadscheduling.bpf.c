#define BPF_NO_GLOBAL_DATA
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

// GPU kernel-shared map (device-visible, mmapable)
#define BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP 1504

// Conservative defaults (should cover most GPUs); out-of-range SM/warp IDs are
// ignored.
#define MAX_SMS 256
#define MAX_WARPS_PER_SM 64

struct {
	__uint(type, BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP);
	__uint(max_entries, MAX_SMS);
	__type(key, u32);
	__type(value, u64);
	__uint(map_flags, BPF_F_MMAPABLE);
} sm_warp_counts SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_GPU_KERNEL_SHARED_ARRAY_MAP);
	__uint(max_entries, MAX_SMS * MAX_WARPS_PER_SM);
	__type(key, u32);
	__type(value, u64);
	__uint(map_flags, BPF_F_MMAPABLE);
} warp_counts SEC(".maps");

static const u64 (*bpf_get_sm_id)(void) = (void *)509;
static const u64 (*bpf_get_warp_id)(void) = (void *)510;
static const u64 (*bpf_get_lane_id)(void) = (void *)511;

SEC("kprobe/placeholder")
int cuda__probe_vllm_threadscheduling()
{
	u64 lane_id = bpf_get_lane_id();
	if (lane_id != 0)
		return 0;

	u64 sm_id = bpf_get_sm_id();
	u64 warp_id = bpf_get_warp_id();
	if (sm_id >= MAX_SMS || warp_id >= MAX_WARPS_PER_SM)
		return 0;

	u32 sm_key = (u32)sm_id;
	u64 *sm_cnt = bpf_map_lookup_elem(&sm_warp_counts, &sm_key);
	if (sm_cnt)
		__atomic_add_fetch(sm_cnt, 1, __ATOMIC_RELAXED);

	u32 warp_key = (u32)(sm_id * MAX_WARPS_PER_SM + warp_id);
	u64 *warp_cnt = bpf_map_lookup_elem(&warp_counts, &warp_key);
	if (warp_cnt)
		__atomic_add_fetch(warp_cnt, 1, __ATOMIC_RELAXED);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";

