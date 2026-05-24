# MoE benchmark report

Run ID: `qwen3-a3b-4gpu-parallelism`
Model: `Qwen/Qwen3-30B-A3B`
GPUs: `4`

## Summary

- Total rows: 54
- Valid rows: 54
- Failed/invalid rows: 0
- Ranked workload winners: 6

## Optimization candidates / server parameters

Each candidate is one backend-specific `server_params_json` vector. Workload fields like input length, output length, and concurrency are constraints measured against these candidates.

| candidate_id | server_state_sha | serve_config | backend |
| --- | --- | --- | --- |
| a14324327f8c | 0675252c4759 | sglang_tp4 | sglang |
| ecde138830b5 | 09a06e21b4f2 | sglang_tp4_dpattn | sglang |
| 982d1b6dddc5 | 90ea3900f2ae | sglang_tp4_dpattn2_ep | sglang |
| 058c3e72a3b9 | 3c06e5615fd5 | sglang_tp4_dpattn_ep | sglang |
| 062e476d673a | 297a61c588cb | sglang_tp4_ep | sglang |
| b74cb14ac0c9 | 96ff1c2efd6b | vllm_tp2dp2 | vllm |
| 44075407cd3e | e7cd2996774d | vllm_tp2dp2_ep | vllm |
| d3ac2d162a80 | 9a0696003dfe | vllm_tp4dp1 | vllm |
| 5caf3fdc02e4 | 702117e71a9e | vllm_tp4dp1_ep | vllm |

### `a14324327f8c` / `sglang_tp4` / `sglang`

```json
{
  "attention_backend": "auto",
  "chunked_prefill_size": "8192",
  "dp": "1",
  "dtype": "bfloat16",
  "ep": "none",
  "max_running_requests": "128",
  "mem_fraction_static": "0.85",
  "moe_a2a_backend": "none",
  "moe_runner_backend": "auto",
  "tp": "4"
}
```

Server command example from run:

```bash
.backends/sglang/bin/python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --host 127.0.0.1 --port 19101 --dtype bfloat16 --trust-remote-code --tp 4 --context-length 8192 --mem-fraction-static 0.85 --disable-radix-cache --max-running-requests 128 --chunked-prefill-size 8192
```

### `ecde138830b5` / `sglang_tp4_dpattn` / `sglang`

```json
{
  "attention_backend": "triton",
  "chunked_prefill_size": "8192",
  "dp": "4",
  "dtype": "bfloat16",
  "ep": "none",
  "max_running_requests": "128",
  "mem_fraction_static": "0.85",
  "moe_a2a_backend": "none",
  "moe_runner_backend": "auto",
  "tp": "4"
}
```

Server command example from run:

```bash
.backends/sglang/bin/python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --host 127.0.0.1 --port 19101 --dtype bfloat16 --trust-remote-code --tp 4 --context-length 8192 --mem-fraction-static 0.85 --disable-radix-cache --max-running-requests 128 --chunked-prefill-size 8192 --attention-backend triton --dp 4 --enable-dp-attention
```

### `982d1b6dddc5` / `sglang_tp4_dpattn2_ep` / `sglang`

```json
{
  "attention_backend": "triton",
  "chunked_prefill_size": "8192",
  "dp": "2",
  "dtype": "bfloat16",
  "ep": "4",
  "max_running_requests": "128",
  "mem_fraction_static": "0.85",
  "moe_a2a_backend": "none",
  "moe_runner_backend": "auto",
  "tp": "4"
}
```

Server command example from run:

```bash
.backends/sglang/bin/python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --host 127.0.0.1 --port 19101 --dtype bfloat16 --trust-remote-code --tp 4 --context-length 8192 --mem-fraction-static 0.85 --disable-radix-cache --max-running-requests 128 --chunked-prefill-size 8192 --attention-backend triton --dp 2 --enable-dp-attention --ep 4 --moe-runner-backend auto --moe-a2a-backend none
```

### `058c3e72a3b9` / `sglang_tp4_dpattn_ep` / `sglang`

```json
{
  "attention_backend": "triton",
  "chunked_prefill_size": "8192",
  "dp": "4",
  "dtype": "bfloat16",
  "ep": "4",
  "max_running_requests": "128",
  "mem_fraction_static": "0.85",
  "moe_a2a_backend": "none",
  "moe_runner_backend": "auto",
  "tp": "4"
}
```

Server command example from run:

```bash
.backends/sglang/bin/python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --host 127.0.0.1 --port 19101 --dtype bfloat16 --trust-remote-code --tp 4 --context-length 8192 --mem-fraction-static 0.85 --disable-radix-cache --max-running-requests 128 --chunked-prefill-size 8192 --attention-backend triton --dp 4 --enable-dp-attention --ep 4 --moe-runner-backend auto --moe-a2a-backend none
```

### `062e476d673a` / `sglang_tp4_ep` / `sglang`

```json
{
  "attention_backend": "flashinfer",
  "chunked_prefill_size": "8192",
  "dp": "1",
  "dtype": "bfloat16",
  "ep": "4",
  "max_running_requests": "128",
  "mem_fraction_static": "0.85",
  "moe_a2a_backend": "none",
  "moe_runner_backend": "auto",
  "tp": "4"
}
```

Server command example from run:

```bash
.backends/sglang/bin/python -m sglang.launch_server --model-path Qwen/Qwen3-30B-A3B --host 127.0.0.1 --port 19101 --dtype bfloat16 --trust-remote-code --tp 4 --context-length 8192 --mem-fraction-static 0.85 --disable-radix-cache --max-running-requests 128 --chunked-prefill-size 8192 --attention-backend flashinfer --ep 4 --moe-runner-backend auto --moe-a2a-backend none
```

### `b74cb14ac0c9` / `vllm_tp2dp2` / `vllm`

```json
{
  "all2all_backend": "auto",
  "data_parallel_size": "2",
  "dtype": "bfloat16",
  "enable_expert_parallel": "False",
  "expert_placement_strategy": "linear",
  "gpu_memory_utilization": "0.85",
  "max_num_batched_tokens": "8192",
  "max_num_seqs": "128",
  "moe_backend": "auto",
  "tensor_parallel_size": "2"
}
```

Server command example from run:

```bash
/mnt/projects/AI/josh/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve Qwen/Qwen3-30B-A3B --served-model-name moe-bench-qwen3-a3b-4gpu-parallelism --host 127.0.0.1 --port 19100 --dtype bfloat16 --trust-remote-code --moe-backend auto --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 128 --max-num-batched-tokens 8192 --tensor-parallel-size 2 --data-parallel-size 2 --no-enable-expert-parallel
```

### `44075407cd3e` / `vllm_tp2dp2_ep` / `vllm`

```json
{
  "all2all_backend": "allgather_reducescatter",
  "data_parallel_size": "2",
  "dtype": "bfloat16",
  "enable_expert_parallel": "True",
  "expert_placement_strategy": "linear",
  "gpu_memory_utilization": "0.85",
  "max_num_batched_tokens": "8192",
  "max_num_seqs": "128",
  "moe_backend": "auto",
  "tensor_parallel_size": "2"
}
```

Server command example from run:

```bash
/mnt/projects/AI/josh/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve Qwen/Qwen3-30B-A3B --served-model-name moe-bench-qwen3-a3b-4gpu-parallelism --host 127.0.0.1 --port 19100 --dtype bfloat16 --trust-remote-code --moe-backend auto --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 128 --max-num-batched-tokens 8192 --tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel --enable-ep-weight-filter --all2all-backend allgather_reducescatter --expert-placement-strategy linear
```

### `d3ac2d162a80` / `vllm_tp4dp1` / `vllm`

```json
{
  "all2all_backend": "auto",
  "data_parallel_size": "1",
  "dtype": "bfloat16",
  "enable_expert_parallel": "False",
  "expert_placement_strategy": "linear",
  "gpu_memory_utilization": "0.85",
  "max_num_batched_tokens": "8192",
  "max_num_seqs": "128",
  "moe_backend": "auto",
  "tensor_parallel_size": "4"
}
```

Server command example from run:

```bash
/mnt/projects/AI/josh/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve Qwen/Qwen3-30B-A3B --served-model-name moe-bench-qwen3-a3b-4gpu-parallelism --host 127.0.0.1 --port 19100 --dtype bfloat16 --trust-remote-code --moe-backend auto --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 128 --max-num-batched-tokens 8192 --tensor-parallel-size 4 --data-parallel-size 1 --no-enable-expert-parallel
```

### `5caf3fdc02e4` / `vllm_tp4dp1_ep` / `vllm`

```json
{
  "all2all_backend": "allgather_reducescatter",
  "data_parallel_size": "1",
  "dtype": "bfloat16",
  "enable_expert_parallel": "True",
  "expert_placement_strategy": "linear",
  "gpu_memory_utilization": "0.85",
  "max_num_batched_tokens": "8192",
  "max_num_seqs": "128",
  "moe_backend": "auto",
  "tensor_parallel_size": "4"
}
```

Server command example from run:

```bash
/mnt/projects/AI/josh/vllm/.venv/bin/python -m vllm.entrypoints.cli.main serve Qwen/Qwen3-30B-A3B --served-model-name moe-bench-qwen3-a3b-4gpu-parallelism --host 127.0.0.1 --port 19100 --dtype bfloat16 --trust-remote-code --moe-backend auto --gpu-memory-utilization 0.85 --no-enable-prefix-caching --max-num-seqs 128 --max-num-batched-tokens 8192 --tensor-parallel-size 4 --data-parallel-size 1 --enable-expert-parallel --enable-ep-weight-filter --all2all-backend allgather_reducescatter --expert-placement-strategy linear
```

## Overall winners

| backend_config | wins |
| --- | --- |
| d3ac2d162a80/vllm/vllm_tp4dp1 | 3 |
| a14324327f8c/sglang/sglang_tp4 | 3 |

## Backend win counts

| backend | wins |
| --- | --- |
| vllm | 3 |
| sglang | 3 |

## Best candidate per workload constraint

| workload | input_len | output_len | max_concurrency | candidate_id | backend | serve_config | objective_value | median_p99_ttft_ms | median_p99_tpot_ms | valid_repeats | failed_repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prompt256_out128_conc16_seed0_rep0 | 256 | 128 | 16 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 170.43826623861185 | 802.9281424276996 | 22.517065397467405 | 1 | 0 |
| prompt256_out128_conc4_seed0_rep0 | 256 | 128 | 4 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 78.5246956910227 | 272.73436966584995 | 12.601658994484122 | 1 | 0 |
| prompt256_out128_conc64_seed0_rep0 | 256 | 128 | 64 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 224.1696550886981 | 1502.022790282499 | 33.837979407357714 | 1 | 0 |
| prompt4096_out128_conc16_seed0_rep0 | 4096 | 128 | 16 | a14324327f8c | sglang | sglang_tp4 | 0.0 | 5597.749780443264 | 121.50324612302181 | 1 | 0 |
| prompt4096_out128_conc4_seed0_rep0 | 4096 | 128 | 4 | a14324327f8c | sglang | sglang_tp4 | 29.344309014257995 | 1715.4564191319514 | 76.6820733798134 | 1 | 0 |
| prompt4096_out128_conc64_seed0_rep0 | 4096 | 128 | 64 | a14324327f8c | sglang | sglang_tp4 | 0.0 | 11692.242448260076 | 310.1604419606017 | 1 | 0 |

The full backend-specific server parameters for each `candidate_id` are listed in the **Optimization candidates / server parameters** section above.

## Candidate detail files

- `candidates.csv`: curated flat table with the top tunable server parameters for each backend, default-filled where omitted, plus `server_cmd`.
- `measurements.csv`: every candidate × workload measurement, including `candidate_id` and `server_state_sha`.

## Top-K candidates per workload

| rank | workload | candidate_id | backend | serve_config | objective_value | relative_to_best | median_p99_ttft_ms | median_p99_tpot_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | prompt256_out128_conc16_seed0_rep0 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 170.43826623861185 | 1.0 | 802.9281424276996 | 22.517065397467405 |
| 2 | prompt256_out128_conc16_seed0_rep0 | 5caf3fdc02e4 | vllm | vllm_tp4dp1_ep | 164.5558948658524 | 0.9654867917717246 | 819.37028267188 | 23.357133197407496 |
| 3 | prompt256_out128_conc16_seed0_rep0 | a14324327f8c | sglang | sglang_tp4 | 128.97611526705728 | 0.7567321477354858 | 447.92505114804953 | 25.977446849392436 |
| 4 | prompt256_out128_conc16_seed0_rep0 | 44075407cd3e | vllm | vllm_tp2dp2_ep | 115.42301338382416 | 0.6772130222342975 | 1106.3405264134053 | 36.29749955183196 |
| 5 | prompt256_out128_conc16_seed0_rep0 | ecde138830b5 | sglang | sglang_tp4_dpattn | 109.6756263028635 | 0.6434917974894132 | 444.4338132219855 | 35.43135051929576 |
| 6 | prompt256_out128_conc16_seed0_rep0 | b74cb14ac0c9 | vllm | vllm_tp2dp2 | 109.441352660763 | 0.642117260847668 | 1201.1119418835733 | 35.846033670085184 |
| 7 | prompt256_out128_conc16_seed0_rep0 | 062e476d673a | sglang | sglang_tp4_ep | 107.4335509263082 | 0.630337032271276 | 936.7933190474287 | 28.037378552799055 |
| 8 | prompt256_out128_conc16_seed0_rep0 | 058c3e72a3b9 | sglang | sglang_tp4_dpattn_ep | 98.33289004522595 | 0.5769413888989043 | 510.8796095917933 | 50.96673918776037 |
| 9 | prompt256_out128_conc16_seed0_rep0 | 982d1b6dddc5 | sglang | sglang_tp4_dpattn2_ep | 83.60697826299673 | 0.4905411214753136 | 561.9358055945486 | 48.446728097151215 |
| 1 | prompt256_out128_conc4_seed0_rep0 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 78.5246956910227 | 1.0 | 272.73436966584995 | 12.601658994484122 |
| 2 | prompt256_out128_conc4_seed0_rep0 | a14324327f8c | sglang | sglang_tp4 | 74.96157584415452 | 0.9546242132425669 | 165.34488553996198 | 15.849671045097672 |
| 3 | prompt256_out128_conc4_seed0_rep0 | 5caf3fdc02e4 | vllm | vllm_tp4dp1_ep | 74.91305507956052 | 0.9540063087200849 | 320.16186884371564 | 13.275686590198985 |
| 4 | prompt256_out128_conc4_seed0_rep0 | 062e476d673a | sglang | sglang_tp4_ep | 68.9420534990037 | 0.8779665160407043 | 171.59278422244824 | 17.234421055523256 |
| 5 | prompt256_out128_conc4_seed0_rep0 | ecde138830b5 | sglang | sglang_tp4_dpattn | 56.907492050526905 | 0.7247082150365192 | 489.581441940973 | 30.736902843463465 |
| 6 | prompt256_out128_conc4_seed0_rep0 | 058c3e72a3b9 | sglang | sglang_tp4_dpattn_ep | 56.52989179657341 | 0.7198995335047973 | 267.77375864214264 | 23.787542289651434 |
| 7 | prompt256_out128_conc4_seed0_rep0 | b74cb14ac0c9 | vllm | vllm_tp2dp2 | 56.20627438120012 | 0.7157783151730936 | 1386.9935305637773 | 21.644829191252835 |
| 8 | prompt256_out128_conc4_seed0_rep0 | 44075407cd3e | vllm | vllm_tp2dp2_ep | 46.31010972100332 | 0.5897521704919858 | 425.4180041141808 | 23.2370056504545 |
| 9 | prompt256_out128_conc4_seed0_rep0 | 982d1b6dddc5 | sglang | sglang_tp4_dpattn2_ep | 43.89529448134581 | 0.5589998674310573 | 454.66604638611904 | 29.753635939236045 |
| 1 | prompt256_out128_conc64_seed0_rep0 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 224.1696550886981 | 1.0 | 1502.022790282499 | 33.837979407357714 |
| 2 | prompt256_out128_conc64_seed0_rep0 | 5caf3fdc02e4 | vllm | vllm_tp4dp1_ep | 218.56418251060015 | 0.9749945077272835 | 1465.4554563586134 | 34.89966639624652 |
| 3 | prompt256_out128_conc64_seed0_rep0 | b74cb14ac0c9 | vllm | vllm_tp2dp2 | 184.3933895142353 | 0.8225617755501994 | 1957.7940334193408 | 41.390725024416675 |
| 4 | prompt256_out128_conc64_seed0_rep0 | 44075407cd3e | vllm | vllm_tp2dp2_ep | 178.3664829163393 | 0.7956763052776448 | 1864.2375497391913 | 41.101630162695194 |
| 5 | prompt256_out128_conc64_seed0_rep0 | ecde138830b5 | sglang | sglang_tp4_dpattn | 178.14052562461654 | 0.7946683307967395 | 691.3012891553808 | 22.49200370037995 |
| 6 | prompt256_out128_conc64_seed0_rep0 | a14324327f8c | sglang | sglang_tp4 | 172.26231652385073 | 0.7684461862409121 | 710.4956264200155 | 24.518367962295528 |
| 7 | prompt256_out128_conc64_seed0_rep0 | 058c3e72a3b9 | sglang | sglang_tp4_dpattn_ep | 172.10792145924685 | 0.767757444205141 | 658.3458429959137 | 23.411984671140093 |
| 8 | prompt256_out128_conc64_seed0_rep0 | 062e476d673a | sglang | sglang_tp4_ep | 164.87384598255264 | 0.7354869057425116 | 712.7642280317377 | 25.482310024489035 |
| 9 | prompt256_out128_conc64_seed0_rep0 | 982d1b6dddc5 | sglang | sglang_tp4_dpattn2_ep | 143.69051294880632 | 0.6409900255766183 | 725.4479354957584 | 29.976836955441197 |
| 1 | prompt4096_out128_conc16_seed0_rep0 | a14324327f8c | sglang | sglang_tp4 | 0.0 |  | 5597.749780443264 | 121.50324612302181 |
| 2 | prompt4096_out128_conc16_seed0_rep0 | ecde138830b5 | sglang | sglang_tp4_dpattn | 0.0 |  | 5099.591908132425 | 106.31217166442762 |
| 3 | prompt4096_out128_conc16_seed0_rep0 | 982d1b6dddc5 | sglang | sglang_tp4_dpattn2_ep | 0.0 |  | 5266.751178809209 | 115.45636603282765 |
| 4 | prompt4096_out128_conc16_seed0_rep0 | 058c3e72a3b9 | sglang | sglang_tp4_dpattn_ep | 0.0 |  | 4766.1458992015105 | 103.43012857170173 |
| 5 | prompt4096_out128_conc16_seed0_rep0 | 062e476d673a | sglang | sglang_tp4_ep | 0.0 |  | 5635.087992921472 | 122.76463384437376 |
| 6 | prompt4096_out128_conc16_seed0_rep0 | b74cb14ac0c9 | vllm | vllm_tp2dp2 | 0.0 |  | 10322.707395217149 | 99.87928229231962 |
| 7 | prompt4096_out128_conc16_seed0_rep0 | 44075407cd3e | vllm | vllm_tp2dp2_ep | 0.0 |  | 9307.429988936055 | 89.96103957757414 |
| 8 | prompt4096_out128_conc16_seed0_rep0 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 0.0 |  | 10665.304132665042 | 94.67551086549868 |
| 9 | prompt4096_out128_conc16_seed0_rep0 | 5caf3fdc02e4 | vllm | vllm_tp4dp1_ep | 0.0 |  | 10476.643310531508 | 93.89857217174492 |
| 1 | prompt4096_out128_conc4_seed0_rep0 | a14324327f8c | sglang | sglang_tp4 | 29.344309014257995 | 1.0 | 1715.4564191319514 | 76.6820733798134 |
| 2 | prompt4096_out128_conc4_seed0_rep0 | ecde138830b5 | sglang | sglang_tp4_dpattn | 29.262615139692546 | 0.997216023232111 | 1552.3387011303566 | 71.14904553710424 |
| 3 | prompt4096_out128_conc4_seed0_rep0 | 058c3e72a3b9 | sglang | sglang_tp4_dpattn_ep | 29.090896902312124 | 0.9913641820012616 | 1446.0963674646337 | 70.71901895136914 |
| 4 | prompt4096_out128_conc4_seed0_rep0 | 062e476d673a | sglang | sglang_tp4_ep | 28.58187570293676 | 0.974017677126056 | 1721.8074434879236 | 77.43871130023847 |
| 5 | prompt4096_out128_conc4_seed0_rep0 | 982d1b6dddc5 | sglang | sglang_tp4_dpattn2_ep | 25.797500598408053 | 0.8791313022867024 | 1630.6199817662127 | 78.51117206311129 |
| 6 | prompt4096_out128_conc4_seed0_rep0 | b74cb14ac0c9 | vllm | vllm_tp2dp2 | 0.0 | 0.0 | 4412.336820380297 | 42.76331825516422 |
| 7 | prompt4096_out128_conc4_seed0_rep0 | 44075407cd3e | vllm | vllm_tp2dp2_ep | 0.0 | 0.0 | 3524.1351056529675 | 39.866000199662956 |
| 8 | prompt4096_out128_conc4_seed0_rep0 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 0.0 | 0.0 | 2808.133585472824 | 26.669004207190977 |
| 9 | prompt4096_out128_conc4_seed0_rep0 | 5caf3fdc02e4 | vllm | vllm_tp4dp1_ep | 0.0 | 0.0 | 2742.676365156658 | 27.247095788227703 |
| 1 | prompt4096_out128_conc64_seed0_rep0 | a14324327f8c | sglang | sglang_tp4 | 0.0 |  | 11692.242448260076 | 310.1604419606017 |
| 2 | prompt4096_out128_conc64_seed0_rep0 | ecde138830b5 | sglang | sglang_tp4_dpattn | 0.0 |  | 10528.976475802483 | 393.5570278528908 |
| 3 | prompt4096_out128_conc64_seed0_rep0 | 982d1b6dddc5 | sglang | sglang_tp4_dpattn2_ep | 0.0 |  | 10790.090425742092 | 457.91782680447113 |
| 4 | prompt4096_out128_conc64_seed0_rep0 | 058c3e72a3b9 | sglang | sglang_tp4_dpattn_ep | 0.0 |  | 9297.450733325677 | 356.3089876047923 |
| 5 | prompt4096_out128_conc64_seed0_rep0 | 062e476d673a | sglang | sglang_tp4_ep | 0.0 |  | 11721.749920560978 | 311.7474511834804 |
| 6 | prompt4096_out128_conc64_seed0_rep0 | b74cb14ac0c9 | vllm | vllm_tp2dp2 | 0.0 |  | 17407.037089429796 | 153.35393551187994 |
| 7 | prompt4096_out128_conc64_seed0_rep0 | 44075407cd3e | vllm | vllm_tp2dp2_ep | 0.0 |  | 19310.316857859725 | 169.31004797714382 |
| 8 | prompt4096_out128_conc64_seed0_rep0 | d3ac2d162a80 | vllm | vllm_tp4dp1 | 0.0 |  | 21338.733640370192 | 180.5436772781912 |
| 9 | prompt4096_out128_conc64_seed0_rep0 | 5caf3fdc02e4 | vllm | vllm_tp4dp1_ep | 0.0 |  | 20874.52029593289 | 176.12857369193625 |

## Parameter sensitivity

| backend | parameter | value | rows | valid_rate | avg_output_tok_s_per_gpu | avg_p99_ttft_ms | avg_p99_tpot_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sglang | attention_backend | auto | 6 | 1.0 | 80.012 | 3388.202 | 95.782 |
| sglang | attention_backend | triton | 18 | 1.0 | 70.254 | 3065.69 | 113.798 |
| sglang | attention_backend | flashinfer | 6 | 1.0 | 73.893 | 3483.299 | 97.117 |
| sglang | dp | 1 | 12 | 1.0 | 76.953 | 3435.751 | 96.45 |
| sglang | dp | 4 | 12 | 1.0 | 74.467 | 2979.41 | 107.359 |
| sglang | dp | 2 | 6 | 1.0 | 61.828 | 3238.252 | 126.677 |
| sglang | ep | none | 12 | 1.0 | 77.804 | 3261.286 | 102.864 |
| sglang | ep | 4 | 18 | 1.0 | 69.687 | 3182.0 | 109.522 |
| vllm | all2all_backend | auto | 12 | 1.0 | 87.06 | 6173.153 | 63.81 |
| vllm | all2all_backend | allgather_reducescatter | 12 | 1.0 | 84.853 | 6019.725 | 64.048 |
| vllm | data_parallel_size | 2 | 12 | 1.0 | 75.468 | 6018.822 | 66.221 |
| vllm | data_parallel_size | 1 | 12 | 1.0 | 96.444 | 6174.057 | 61.638 |
| vllm | enable_expert_parallel | False | 12 | 1.0 | 87.06 | 6173.153 | 63.81 |
| vllm | enable_expert_parallel | True | 12 | 1.0 | 84.853 | 6019.725 | 64.048 |
| vllm | tensor_parallel_size | 2 | 12 | 1.0 | 75.468 | 6018.822 | 66.221 |
| vllm | tensor_parallel_size | 4 | 12 | 1.0 | 96.444 | 6174.057 | 61.638 |

## Failure analysis

_No failures recorded._

## Interactive report

Open [`report.html`](report.html) for the interactive view with constraint sliders, Pareto explorer, and per-candidate drill-down.

## Plots

_No plots generated._
