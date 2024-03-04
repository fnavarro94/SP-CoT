export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks
export METHOD=manual-cot
export SLEEP=100
export OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
export OPENAI_API_BASE=https://api.openai.com/v1
export MODEL=text-davinci-003

#export FLAG=random-hotpot
#
#
#export METHOD=zeroshot
#export SLEEP=500
#export DATASET=grail-qa
##export CONTEXT_FILE=demos/$DATASET/$MODEL/davinci_auto-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=100 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --on_test_data
#
#
#export METHOD=zeroshot-cot
#export SLEEP=400
#export DATASET=grail-qa
##export CONTEXT_FILE=demos/$DATASET/$MODEL/davinci_auto-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=100 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --on_test_data \
#    --use_cache
#
#
#export METHOD=self-prompt-cot
#export SLEEP=90
#export DATASET=grail-qa
#export CONTEXT_FILE=demos/$DATASET/gpt-3.5-turbo-0301/self-prompt-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=100 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --input_file=$CONTEXT_FILE \
#    --on_test_data \
#    --use_cache


export METHOD=manual-cot
export SLEEP=190
export DATASET=grail-qa
export CONTEXT_FILE=demos/manual-cot-random-musique.json
python $TASK_PATH/inference_async.py \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --answer_tokens=20 \
    --do_eval \
    --max_concurrency=50 \
    --limit_dataset_size=0 \
    --sleep_every_n_requests=$SLEEP \
    --manual_cot_path=$CONTEXT_FILE \
    --on_test_data \
    --flag=random-musique



export METHOD=manual-cot
export SLEEP=190
export DATASET=grail-qa
export CONTEXT_FILE=demos/manual-cot-random-hotpot.json
python $TASK_PATH/inference_async.py \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --answer_tokens=20 \
    --do_eval \
    --max_concurrency=50 \
    --limit_dataset_size=0 \
    --sleep_every_n_requests=$SLEEP \
    --manual_cot_path=$CONTEXT_FILE \
    --on_test_data \
    --flag=random-hotpot


#export METHOD=genread
#export SLEEP=50
#export DATASET=grail-qa
#export CONTEXT_FILE=demos/manual-cot.json
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --manual_cot_path=$CONTEXT_FILE \
#    --on_test_data \
#    --use_cache


#export CONTEXT_FILE=demos/manual-cot-random-musique.json
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --on_test_data

#export CONTEXT_FILE=demos/manual-cot-random-hotpot.json
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --manual_cot_path=$CONTEXT_FILE \
#    --on_test_data \
#    --flag=random-hotpot \
#    --use_cache














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
#
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
#






## zeroshot
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
#    --flag=$FLAG \
#    --on_test_data
#
#export DATASET=hotpot-qa
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --on_test_data
#
#export DATASET=musique-qa
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=100 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --on_test_data
#
#export DATASET=wikimh-qa
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=$MODEL \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=100 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --on_test_data





## self prompt cot
#export DATASET=cweb-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=text-davinci-003 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --on_test_data

#export DATASET=hotpot-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=text-davinci-003 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --on_test_data
#
#
#export DATASET=musique-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=text-davinci-003 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --on_test_data
#
#
#
#export DATASET=wikimh-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_retrieve_in_cluster_demos-8_test.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=text-davinci-003 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --on_test_data


















#export PYTHONPATH=$PYTHONPATH:../
#export TASK_PATH=tasks
#export METHOD=auto-cot
#export SLEEP=50
#
#
#
## Auto Cot
#export DATASET=cweb-qa
#export DATA_PATH=data/$DATASET/auto-cot_train.json
#export DEMO_PATH=demos/$DATASET/auto-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=30 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=auto-cot
#
#export DATASET=hotpot-qa
#export DATA_PATH=data/$DATASET/auto-cot_train.json
#export DEMO_PATH=demos/$DATASET/auto-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=30 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=auto-cot
#
#export DATASET=musique-qa
#export DATA_PATH=data/$DATASET/auto-cot_train.json
#export DEMO_PATH=demos/$DATASET/auto-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=30 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=auto-cot
#
#export DATASET=wikimh-qa
#export DATA_PATH=data/$DATASET/auto-cot_train.json
#export DEMO_PATH=demos/$DATASET/auto-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=30 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=auto-cot
##






#export PYTHONPATH=$PYTHONPATH:../
#export TASK_PATH=tasks
#export METHOD=self-prompt-cot
#export SLEEP=40
#export FLAG=knowledge-only
#
#
## self prompt cot
#export DATASET=cweb-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only_self-prompt-cot_retrieve_in_cluster_demos-8.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --use_cache
#
#
#export DATASET=hotpot-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only_self-prompt-cot_retrieve_in_cluster_demos-8.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --use_cache
#
#
#export DATASET=musique-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only_self-prompt-cot_retrieve_in_cluster_demos-8.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --use_cache
#
#
#
#export DATASET=wikimh-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only_self-prompt-cot_retrieve_in_cluster_demos-8.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=$FLAG \
#    --use_cache
#







#export DATASET=wikimh-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=knowledge-only-test
#
#export DATASET=musique-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=knowledge-only-test
#
#export DATASET=cweb-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=knowledge-only-test
#
#















#export DATASET=hotpot-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=knowledge-only-test
#
#
#export DATASET=wikimh-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=knowledge-only-test
#
#export DATASET=musique-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=knowledge-only-test
#
#export DATASET=cweb-qa
#export DEMO_PATH=demos/$DATASET/knowledge-only-test_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=knowledge-only-test








##
#export DATASET=musique-qa
#export DEMO_PATH=demos/$DATASET/qa_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=30 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=qa_only \
#    --use_cache
##
#export DATASET=complex-qa
#export DEMO_PATH=demos/$DATASET/qa_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=30 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=qa_only \
#    --use_cache
##
##
#export DATASET=cweb-qa
#export DEMO_PATH=demos/$DATASET/qa_self-prompt-cot_c-8_Q.json
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=30 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=qa_only \
#    --use_cache



#
#
#
#export DATASET=wikimh-qa
#export DEMO_PATH=demos/$DATASET/self-prompt_cluster-8_content-Q.json
#
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=20 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP

#
#export METHOD=self-prompt-cot
#export DATASET=complex-qa
#export DEMO_PATH=demos/$DATASET/qa_self-prompt-cot_c-6_Q.json
#export SLEEP=25
###
#python $TASK_PATH/inference_async.py \
#    --input_file=$DEMO_PATH \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --flag=prompt_test_v1_all
#
#
#
#




#export DATASET=cweb-qa
##
#python $TASK_PATH/inference_async.py \
#    --input_file=demos/$DATASET/"$METHOD"_cluster-3_content-Q_k-5.json \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --limit_dataset_size=20 \
#    --flag=c_3_k_5



#export DATASET=complex-qa
##
#python $TASK_PATH/inference_async.py \
#    --input_file=demos/complex-qa/self-prompt-cot_cluster-3_content-Q_k-5.json \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=10 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --limit_dataset_size=20 \
#    --flag=c_3_k_5





#export DATASET=cweb-qa
##
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --limit_dataset_size=0 \
#    --use_cache
#
##
#export DATASET=complex-qa
##
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --limit_dataset_size=0 \
#    --use_cache
##
##
##
##
#
#export DATASET=musique-qa
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --use_cache
#

#
#
#
#export DATASET=hotpot-qa
#
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --use_cache


#export DATASET=grail-qa
#
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --use_cache
#
#export DATASET=hybrid-qa
#
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP \
#    --use_cache
#

#export DATASET=musique-qa
#
#python $TASK_PATH/inference_async.py \
#    --task=$DATASET \
#    --method=$METHOD \
#    --model_name=gpt-3.5-turbo-0301 \
#    --answer_tokens=20 \
#    --do_eval \
#    --max_concurrency=50 \
#    --limit_dataset_size=0 \
#    --sleep_every_n_requests=$SLEEP
##
