export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks
export METHOD=self-prompt-cot
export SLEEP=40


export MODEL=gpt-3.5-turbo-0301
#export OPENAI_API_KEY=sk-fluQKnfYDVHPYcWiBeA6C6C18b1d44709e870454B4B941F4
#export OPENAI_API_BASE=https://1api.onekey.asia/v1
export OPENAI_API_KEY=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
export OPENAI_API_BASE=https://api.openai.com/v1


export DATASET=cweb-qa
export DEMO_PATH=demos/$DATASET/knowledge-only_self-prompt-cot_retrieve_in_cluster_demos-8.json
python $TASK_PATH/self_consistency_async.py \
    --task=$DATASET \
    --input_file=$DEMO_PATH \
    --method=$METHOD \
    --model_name=$MODEL \
    --answer_tokens=20 \
    --do_eval \
    --max_concurrency=50 \
    --limit_dataset_size=0 \
    --sleep_every_n_requests=$SLEEP \
    --use_cache


#export DATASET=musique-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only_self-prompt-cot_retrieve_in_cluster_demos-8.json
#python $TASK_PATH/self_consistency_async.py \
#    --task=$DATASET \
#    --input_file=$DEMO_PATH \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --use_cache



#export DATASET=hotpot-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/self_consistency_async.py \
#    --task=$DATASET \
#    --input_file=$DEMO_PATH \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP






























## Manual CoT

#export DATASET=musique-qa
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --manual_cot_path=demos/manual-cot-random-hotpot.json \
#    --flag=$FLAG
#
#
#export DATASET=cweb-qa
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --manual_cot_path=demos/manual-cot-random-hotpot.json \
#    --flag=$FLAG
#
##
#export DATASET=hotpot-qa
#python $TASK_PATH/self_consistency_async.py \
#    --task=$DATASET \
#    --input_file=$DEMO_PATH \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=10 \
#    --sleep_every_n_requests=$SLEEP \
#    --manual_cot_path=demos/manual-cot-random-hotpot.json \
#    --flag=$FLAG


#export DATASET=wikimh-qa
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --manual_cot_path=demos/manual-cot-random-hotpot.json \
#    --flag=$FLAG \
#    --use_cache

#

