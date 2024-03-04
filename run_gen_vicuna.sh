export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks/gen_qa_pairs_vicuna.py
export MODEL=vicuna-13b
export CUDA_VISIBLE_DEVICES=0

python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=0

