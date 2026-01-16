ROOT=ExGRPO
TEMPLATE=own
MODEL_PATH=$ROOT/exgrpo
OUTPUT_DIR=./results/

DATA=$ROOT/data/valid.id.parquet
MODEL_NAME=exgrpo+testid

mkdir -p $OUTPUT_DIR

python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log

DATA=$ROOT/data/valid.ood.parquet
MODEL_NAME=exgrpo+testood

mkdir -p $OUTPUT_DIR

python generate_vllm.py \
   --model_path $MODEL_PATH \
   --input_file $DATA \
   --remove_system True \
   --add_oat_evaluate True \
   --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
   --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
