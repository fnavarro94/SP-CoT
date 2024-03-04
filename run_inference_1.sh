export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks
export METHOD=zeroshot
export MODEL=wizard-13b
export CUDA_VISIBLE_DEVICES=0
export BATCH_SIZE=1
export HF_DATASETS_CACHE="/data2/.cache/huggingface/datasets"
#export FLAG=auto-cot-step1-train-1k

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

export DATASET=hotpot-qa
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

#export DATASET=musique-qa
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
#export PYTHONPATH=$PYTHONPATH:../
#export TASK_PATH=tasks
#export METHOD=self-prompt-cot
#export MODEL=falcon-7b
#export CUDA_VISIBLE_DEVICES=0
#export BATCH_SIZE=1
##
##
### self prompt cot
##export DEMO_NAME=knowledge-only-test_self-prompt-cot_retrieve_in_cluster_demos-8_test.json
#export DEMO_NAME=falcon_self-prompt-cot_retrieve_in_cluster_demos-6_test.json
#export FLAG=falcon_demo_6
#export COT_TOKENS=100
#
#export DATASET=hotpot-qa
#export DEMO_PATH=demos/$DATASET/$MODEL/$DEMO_NAME
##export DEMO_PATH=demos/$DATASET/$DEMO_NAME
#python $TASK_PATH/inference_batch.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=0 \
#    --batch_size=$BATCH_SIZE  \
#    --flag=$FLAG \
#    --cot_tokens=$COT_TOKENS \
#    --num_demos=6
####
###
##export DATASET=cweb-qa
##export DEMO_PATH=demos/$DATASET/$MODEL/$DEMO_NAME
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
##    --cot_tokens=$COT_TOKENS
##
#
#
#export DATASET=hotpot-qa
#export DEMO_PATH=demos/$DATASET/$MODEL/$DEMO_NAME
##export DEMO_PATH=demos/$DATASET/$DEMO_NAME
#
#python $TASK_PATH/inference_batch.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=40 \
#    --batch_size=$BATCH_SIZE \
#    --flag=$FLAG \
#    --cot_tokens=$COT_TOKENS


#export DATASET=musique-qa
#export DEMO_PATH=demos/$DATASET/$MODEL/$DEMO_NAME
##export DEMO_PATH=demos/$DATASET/$DEMO_NAME
#
#python $TASK_PATH/inference_batch.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --do_eval \
#    --limit_dataset_size=40 \
#    --batch_size=$BATCH_SIZE \
#    --flag=$FLAG \
#    --cot_tokens=$COT_TOKENS

