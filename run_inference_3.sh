export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks
export METHOD=zeroshot
export MODEL=wizard-13b
export CUDA_VISIBLE_DEVICES=2
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

export DATASET=wikimh-qa
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
