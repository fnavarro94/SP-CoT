export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks
export METHOD=zeroshot
export MODEL=vicuna-13b
export CUDA_VISIBLE_DEVICES=3
export BATCH_SIZE=1
export HF_DATASETS_CACHE="/data2/.cache/huggingface/datasets"
export FLAG=auto-cot-step1-train-1k

export FLAG=dpr-retrieval

#export DATASET=cweb-qa
#export CONTEXT_PATH=data/$DATASET/dpr/retrieval_only-dpr.json
#python $TASK_PATH/inference_batch.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=0 \
#    --batch_size=$BATCH_SIZE \
#    --input_file=$CONTEXT_PATH \
#    --flag=$FLAG
#
export DATASET=cweb-qa
export CONTEXT_PATH=data/$DATASET/dpr/retrieval_only-dpr.json
python $TASK_PATH/inference_batch.py \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --do_eval \
    --limit_dataset_size=0 \
    --batch_size=$BATCH_SIZE \
    --input_file=$CONTEXT_PATH \
    --flag=$FLAG

export DATASET=musique-qa
export CONTEXT_PATH=data/$DATASET/dpr/retrieval_only-dpr.json
python $TASK_PATH/inference_batch.py \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --do_eval \
    --limit_dataset_size=0 \
    --batch_size=$BATCH_SIZE \
    --input_file=$CONTEXT_PATH \
    --flag=$FLAG













#export DATASET=cweb-qa
#
#python $TASK_PATH/inference_batch.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=0 \
#    --batch_size=$BATCH_SIZE \
#    --on_test_data
#
##
#export DATASET=wikimh-qa
#python $TASK_PATH/inference_batch.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=0 \
#    --batch_size=$BATCH_SIZE \
#    --on_test_data

##
#export DATASET=hotpot-qa
#export STEP1_PATH=demos/$DATASET/$MODEL/wizard_auto-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_batch.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=1000 \
#    --batch_size=$BATCH_SIZE \
#    --on_test_data \
#    --input_file=$STEP1_PATH

#export DATASET=musique-qa
#export STEP1_PATH=demos/$DATASET/$MODEL/wizard_auto-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_batch.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=1000 \
#    --batch_size=$BATCH_SIZE \
#    --on_test_data \
#    --input_file=$STEP1_PATH
#






















#export PYTHONPATH=$PYTHONPATH:../
#export TASK_PATH=tasks
#export METHOD=self-prompt-cot
#export MODEL=falcon-7b
#export CUDA_VISIBLE_DEVICES=5
#export BATCH_SIZE=1
#
##export DEMO_NAME=wizard_self-prompt-cot_retrieve_in_cluster_demos-8_test.json
##
##export DATASET=hotpot-qa
##export DEMO_PATH=demos/$DATASET/$MODEL/$DEMO_NAME
##export FLAG=wizard_demo_8
##
##python $TASK_PATH/inference_batch.py \
##    --input_file=$DEMO_PATH \
##    --task=$DATASET \
##    --method=$METHOD \
##    --model_name=$MODEL \
##    --do_eval \
##    --limit_dataset_size=0 \
##    --batch_size=$BATCH_SIZE \
##    --flag=$FLAG \
##    --num_demos=8
#
#
#export DEMO_NAME=falcon_self-prompt-cot_retrieve_in_cluster_demos-4_test.json
#
#export DATASET=hotpot-qa
#export DEMO_PATH=demos/$DATASET/$MODEL/$DEMO_NAME
#export FLAG=falcon_demo_4
#
#python $TASK_PATH/inference_batch.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=0 \
#    --batch_size=$BATCH_SIZE \
#    --flag=$FLAG \
#    --num_demos=4 \
#    --cot_tokens=100
