// vLLM SM/Warp/Lane probe loader (CUDA kprobe)
//
// Attaches a CUDA eBPF program to a selected kernel symbol and reports
// approximate SM/warp distribution. The probe runs only on lane0 of each warp
// to keep overhead manageable for large kernels.

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "./.output/vllm_threadscheduling.skel.h"

#define warn(...) fprintf(stderr, __VA_ARGS__)

#define MAX_SMS 256
#define MAX_WARPS_PER_SM 64

static int libbpf_print_fn(enum libbpf_print_level level, const char *format,
			   va_list args)
{
	return vfprintf(stderr, format, args);
}

static volatile bool exiting = false;

static void sig_handler(int sig)
{
	(void)sig;
	exiting = true;
}

static void usage(const char *argv0)
{
	printf("Usage: %s --func <CUDA_KERNEL_SYMBOL_NAME> [--interval <sec>]\n",
	       argv0);
	printf("\n");
	printf("Examples:\n");
	printf("  %s --func '_ZN2at6native29vectorized_elementwise_kernel...'\n",
	       argv0);
}

static void print_stats(struct vllm_threadscheduling_bpf *skel)
{
	const int sm_fd = bpf_map__fd(skel->maps.sm_warp_counts);
	const int warp_fd = bpf_map__fd(skel->maps.warp_counts);

	uint64_t sm_counts[MAX_SMS] = { 0 };
	uint64_t total = 0;
	uint64_t max_count = 0;
	int active_sms = 0;

	for (uint32_t sm = 0; sm < MAX_SMS; sm++) {
		uint64_t v = 0;
		if (bpf_map_lookup_elem(sm_fd, &sm, &v) == 0 && v > 0) {
			sm_counts[sm] = v;
			total += v;
			if (v > max_count)
				max_count = v;
			active_sms++;
		}
	}

	// Clear screen (best-effort)
	printf("\033[2J\033[H");

	time_t t;
	time(&t);
	struct tm *tm = localtime(&t);
	char ts[32] = { 0 };
	if (tm)
		strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", tm);

	printf("vLLM SM/Warp probe\n");
	printf("Timestamp: %s\n\n", ts[0] ? ts : "(unknown)");

	if (total == 0) {
		printf("No data collected yet.\n");
		fflush(stdout);
		return;
	}

	printf("Total sampled warps: %lu  Active SMs (seen): %d\n\n", total,
	       active_sms);

	// Simple histogram
	const int bar_width = 40;
	printf("SM distribution (sampled warps per SM)\n");
	for (int sm = 0; sm < MAX_SMS; sm++) {
		if (sm_counts[sm] == 0)
			continue;
		int bar_len = (max_count > 0) ?
				      (int)((sm_counts[sm] * bar_width) /
					    max_count) :
				      0;
		if (bar_len == 0)
			bar_len = 1;
		printf("  SM %3d: ", sm);
		for (int j = 0; j < bar_len; j++)
			printf("█");
		for (int j = bar_len; j < bar_width; j++)
			printf(" ");
		printf(" %8lu\n", sm_counts[sm]);
	}

	printf("\nWarp slots (first 20 non-zero):\n");
	int shown = 0;
	for (uint32_t sm = 0; sm < MAX_SMS && shown < 20; sm++) {
		for (uint32_t w = 0; w < MAX_WARPS_PER_SM && shown < 20; w++) {
			uint32_t key = sm * MAX_WARPS_PER_SM + w;
			uint64_t v = 0;
			if (bpf_map_lookup_elem(warp_fd, &key, &v) == 0 &&
			    v > 0) {
				printf("  SM %3u warp %2u: %lu\n", sm, w, v);
				shown++;
			}
		}
	}
	if (shown == 0)
		printf("  (none)\n");

	printf("\nPress Ctrl+C to exit.\n");
	fflush(stdout);
}

int main(int argc, char **argv)
{
	const char *func = NULL;
	int interval_s = 2;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "--func") == 0 ||
		     strcmp(argv[i], "--kernel") == 0) &&
		    i + 1 < argc) {
			func = argv[i + 1];
			i++;
			continue;
		}
		if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc) {
			interval_s = atoi(argv[i + 1]);
			if (interval_s <= 0)
				interval_s = 2;
			i++;
			continue;
		}
		if (strcmp(argv[i], "-h") == 0 ||
		    strcmp(argv[i], "--help") == 0) {
			usage(argv[0]);
			return 0;
		}
	}

	if (!func || func[0] == '\0') {
		usage(argv[0]);
		return 2;
	}

	libbpf_set_print(libbpf_print_fn);
	signal(SIGINT, sig_handler);
	signal(SIGTERM, sig_handler);

	struct vllm_threadscheduling_bpf *skel =
		vllm_threadscheduling_bpf__open();
	if (!skel) {
		warn("Failed to open BPF skeleton\n");
		return 1;
	}

	int err = vllm_threadscheduling_bpf__load(skel);
	if (err) {
		warn("Failed to load BPF skeleton: %d\n", err);
		goto cleanup;
	}

	struct bpf_link *link = bpf_program__attach_kprobe(
		skel->progs.cuda__probe_vllm_threadscheduling, false, func);
	if (!link) {
		err = -errno;
		warn("Failed to attach CUDA kprobe to '%s': %s\n", func,
		     strerror(errno));
		goto cleanup;
	}

	printf("Attached CUDA probe to: %s\n", func);
	fflush(stdout);

	while (!exiting) {
		sleep(interval_s);
		print_stats(skel);
	}

	bpf_link__destroy(link);

cleanup:
	vllm_threadscheduling_bpf__destroy(skel);
	return err < 0 ? -err : 0;
}

