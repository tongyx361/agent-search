#! /bin/bash
set -e
# gpu_ids=$1
# top_p=$2
# temperatures=("$2")
# temperatures=(0.4 0.7 1.0)
# temperatures=(0.7)
# code_temperatures=(0.7 0.6 0.4)
# code_temperatures=(0.7)
# code_temperatures=("$2")
# init_temperatures=(0.1 0.2 0.3)
# init_temperatures=(0.4 0.5 0.6)
# init_temperatures=(0.7 0.8 0.9)
# init_temperatures=(1.0 1.1 1.2)
# temperatures=(0.6 0.7 0.8)
# temperatures=(0.2 0.3 0.4 0.5 0.6)
# init_temperatures=(0.6 0.7 0.8)
# init_temperatures=(0.8 0.9 1.0)
# init_temperatures=(1.0)

# for temperature in "${temperatures[@]}"; do
# for code_temperature in "${code_temperatures[@]}"; do
# for init_temperature in "${init_temperatures[@]}"; do
# if [ "${temperature}" == "0.4" ] && [ "${code_temperature}" == "0.4" ] &&
#     { [ "${init_temperature}" == "0.4" ] || [ "${init_temperature}" == "1.0" ]; }; then
#     continue # Already done
# fi
gpu_ids_list=(2 3)
# temperatures=(0 0.3 0.6)
temperatures=(0.3 0.6)
# temperature=0.6
init_temperature=1.0
top_p=0.95
# top_ps=(0.9 1.0)
log_dir="logs/it1.0-long-instr"
mkdir -p "${log_dir}"
for idx in "${!gpu_ids_list[@]}"; do
    gpu_ids="${gpu_ids_list[$idx]}"
    temperature="${temperatures[$idx]}"
    # top_p="${top_ps[$idx]}"
    # code_temperature=0.7
    CUDA_VISIBLE_DEVICES="${gpu_ids}" python pipelines/search.py \
        --temperature "${temperature}" \
        --code_temperature "${temperature}" \
        --init_temperature "${init_temperature}" \
        --top_p "${top_p}" \
        --allow_empty_output \
        --allow_timeout \
        --allow_err \
        --seed 42 \
        --gen_dset_fpath data/balanced_hardest_int_ans_math.json \
        >"${log_dir}/t${temperature}.log" 2>&1 &
    sleep 5
done
wait
# --allow_empty_output \
# done
# done
# done
