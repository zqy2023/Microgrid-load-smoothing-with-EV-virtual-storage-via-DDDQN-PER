#!/usr/bin/env bash

# =============================================================
# EVVES-DDDQN 批量训练脚本 (半年级联)
# -------------------------------------------------------------
# 说明：
#   - 按随机种子顺序训练 DDDQN-PER 模型；
#   - 训练完成后模型保存在指定目录；
#   - 所有参数可通过环境变量覆盖。
# =============================================================

set -euo pipefail

# ----------------- 参数 -----------------
SEEDS=${SEEDS:-"0 1 2 3"}
STEPS=${STEPS:-1200000}
NUM_ENVS=${NUM_ENVS:-8}
BUFFER_SIZE=${BUFFER_SIZE:-200000}
NET_WIDTH=${NET_WIDTH:-256}
OUTPUT_DIR=${OUTPUT_DIR:-"models"}

DATA_LOAD=${DATA_LOAD:-"data_processed/load_15min_fullyear.parquet"}
DATA_PRICE=${DATA_PRICE:-"data_processed/price_15min_fullyear.parquet"}
EV_DEMAND=${EV_DEMAND:-"data_processed/ev_demand_15min_fullyear.parquet"}
SOH_JSON=${SOH_JSON:-"data_processed/soh_params_fullyear.json"}
# ----------------------------------------

mkdir -p "${OUTPUT_DIR}"

# 配置摘要
cat <<CFG
=============================================================
Seeds        : ${SEEDS}
Steps        : ${STEPS}
Vec Envs     : ${NUM_ENVS}
Buffer Size  : ${BUFFER_SIZE}
Net Width    : ${NET_WIDTH}
Output Dir   : ${OUTPUT_DIR}
-------------------------------------------------------------
Load Data    : ${DATA_LOAD}
Price Data   : ${DATA_PRICE}
EV Demand    : ${EV_DEMAND}
SOH Params   : ${SOH_JSON}
=============================================================
CFG

# 数据完整性检查
for file in "${DATA_LOAD}" "${DATA_PRICE}" "${EV_DEMAND}" "${SOH_JSON}"; do
  [[ -f "$file" ]] || { echo "缺少数据文件：$file" >&2; exit 1; }
done

# 主循环
for SEED in ${SEEDS}; do
  echo ""
  echo ">>> 开始训练 SEED=${SEED}"
  python code/train_dddqn.py \
    --load "${DATA_LOAD}" \
    --price "${DATA_PRICE}" \
    --ev "${EV_DEMAND}" \
    --soh "${SOH_JSON}" \
    --save "${OUTPUT_DIR}/dddqn_per_seed_${SEED}.zip" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --algorithm dddqn \
    --vec_env "${NUM_ENVS}" \
    --buffer "${BUFFER_SIZE}" \
    --net_width "${NET_WIDTH}"
  echo "<<< SEED=${SEED} 训练完成。"
  echo "---------------------------------------"
  sleep 2
done

echo "全部训练完成，模型保存在 ${OUTPUT_DIR}/"
 
 
 
 
 
 
 
 
 
 