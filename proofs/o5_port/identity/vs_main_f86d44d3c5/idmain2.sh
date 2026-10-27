#!/usr/bin/env bash
# OFF identity of lane/o5-port da16d568b6 (facade + origin/main f86d44d3c5) vs main f86d44d3c5, schema-2 probe, box CPU (o5-port, manager 01:12Z)
source ~/src/wrf_gpu2_env/venv/bin/activate
unset PYTHONPATH
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= GPUWRF_JAX_CACHE=0
cd ~/o5port; mkdir -p idmain2; A=~/o5port/alias_id
B=~/src/o5_base_main_f86d; K=~/src/o5_cand_head_da16
P=$K/scripts/off_identity_probe.py; C=$K/scripts/off_identity_compare.py
PROD=~/o5port/cases/prod_20260725; SW=$K/examples/switzerland_d01
run() { ln -sfn "$2" $A; nice -n 5 timeout 3000 python $P --alias $A --case "$3" --out idmain2/$1.json "${@:4}" > idmain2/$1.log 2>&1; echo "$1 rc=$? $(date -u +%H:%M:%SZ)"; }
run base_prod_nested $B $PROD --nested
run cand_prod_nested $K $PROD --nested
run base_swiss $B $SW
run cand_swiss $K $SW
python $C idmain2/base_prod_nested.json idmain2/cand_prod_nested.json idmain2/ID_prod_nested.json; echo "cmp prod rc=$?"
python $C idmain2/base_swiss.json idmain2/cand_swiss.json idmain2/ID_swiss.json; echo "cmp swiss rc=$?"
echo IDMAIN2_DONE
