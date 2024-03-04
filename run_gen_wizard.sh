export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks/gen_qa_pairs_wizard.py
export MODEL=wizard-13b
export CUDA_VISIBLE_DEVICES=4

python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=0
##
python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=1

python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=2
##
python $TASK_PATH \
    --model_name=$MODEL \
    --topic_index=3
