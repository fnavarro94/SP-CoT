export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks
export CUDA_VISIBLE_DEVICES=7

python $TASK_PATH/gen_qa_pairs_alpaca.py \
    --model_name=alpaca-13b \
    --passage_tokens=300 \
    --topic_index=0

