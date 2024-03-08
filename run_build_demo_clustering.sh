export PYTHONPATH=$PYTHONPATH:../
export TASK_PATH=tasks/build_demo_clustering.py

export MODEL=gpt-3.5-turbo-0301
export CUDA_VISIBLE_DEVICES=7

export NUM_CLUSTERS=8
export SAMPLING=retrieve_in_cluster
## self prompt cot
export METHOD=self-prompt-cot




export DATASET=grail-qa
export GEN_FILE=data/$METHOD/pseudo_dataset.json
python $TASK_PATH \
    --model_name=$MODEL \
    --task=$DATASET \
    --input_gen_file=$GEN_FILE \
    --num_clusters=$NUM_CLUSTERS \
    --sampling=$SAMPLING \
    --method=$METHOD \
    --do_test \
    --limit_gen_size=0










#export FLAG=chat

#export DATASET=cweb-qa
#export GEN_FILE=data/"$DATASET"/"$MODEL"/zeroshot-cot-auto-cot.json
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=$DATASET \
#    --input_gen_file=$GEN_FILE \
#    --num_clusters=$NUM_CLUSTERS \
#    --sampling=$SAMPLING \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test \
#    --limit_gen_size=500
#
#
#export DATASET=musique-qa
#export GEN_FILE=data/"$DATASET"/"$MODEL"/zeroshot-cot-auto-cot.json
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=$DATASET \
#    --input_gen_file=$GEN_FILE \
#    --num_clusters=$NUM_CLUSTERS \
#    --sampling=$SAMPLING \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test \
#    --limit_gen_size=500
#
#
#export DATASET=wikimh-qa
#export GEN_FILE=data/"$DATASET"/"$MODEL"/zeroshot-cot-auto-cot.json
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=$DATASET \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test \
#    --limit_gen_size=500
#
#export DATASET=hotpot-qa
#export GEN_FILE=data/"$DATASET"/"$MODEL"/zeroshot-cot-auto-cot.json
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=$DATASET \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test \
#    --limit_gen_size=500
























#export MODEL=falcon-7b
#export CUDA_VISIBLE_DEVICES=7
#
#export NUM_CLUSTERS=8
#export SAMPLING=retrieve_in_cluster
### self prompt cot
#export METHOD=self-prompt-cot
#export GEN_FILE=data/$METHOD/pseudo_dataset_falcon.json
#export FLAG=falcon
#
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=cweb-qa \
#    --input_gen_file=$GEN_FILE \
#    --num_clusters=$NUM_CLUSTERS \
#    --sampling=$SAMPLING \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=musique-qa \
#    --input_gen_file=$GEN_FILE \
#    --num_clusters=$NUM_CLUSTERS \
#    --sampling=$SAMPLING \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=wikimh-qa \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=hotpot-qa \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test










































#export PYTHONPATH=$PYTHONPATH:../
#export TASK_PATH=tasks/build_demo_clustering.py
#
#export MODEL=gpt-neoxt-chat
#export CUDA_VISIBLE_DEVICES=7
#
#export NUM_CLUSTERS=6
#export SAMPLING=retrieve_in_cluster
### self prompt cot
#export METHOD=self-prompt-cot
#export GEN_FILE=data/$METHOD/pseudo_dataset_neoxt.json
#export FLAG=neoxt
#
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=cweb-qa \
#    --input_gen_file=$GEN_FILE \
#    --num_clusters=$NUM_CLUSTERS \
#    --sampling=$SAMPLING \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=musique-qa \
#    --input_gen_file=$GEN_FILE \
#    --num_clusters=$NUM_CLUSTERS \
#    --sampling=$SAMPLING \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=wikimh-qa \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test
#
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=hotpot-qa \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --do_test
##
#
#







#export PYTHONPATH=$PYTHONPATH:../
#export TASK_PATH=tasks/build_demo_clustering.py
#
#export MODEL=alpaca-13b
#export CUDA_VISIBLE_DEVICES=6
#
#export SAMPLING=retrieve_in_cluster
### self prompt cot
#export METHOD=self-prompt-cot
#export GEN_FILE=data/$METHOD/pseudo_dataset_alpaca.json
#export FLAG=alpaca-dev1k
#
#
#export NUM_CLUSTERS=8
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=wikimh-qa \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --limit_dataset_size=1000 \
#
#export NUM_CLUSTERS=6
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=wikimh-qa \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --limit_dataset_size=1000 \
#
#export NUM_CLUSTERS=3
#python $TASK_PATH \
#    --model_name=$MODEL \
#    --task=wikimh-qa \
#    --input_gen_file=$GEN_FILE \
#    --sampling=$SAMPLING \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --flag=$FLAG \
#    --limit_dataset_size=1000 \






















#export METHOD=auto-cot
#export NUM_CLUSTERS=8
#
#python $TASK_PATH \
#    --task=musique-qa \
#    --input_gen_file=data/musique-qa/$MODEL/zeroshot-cot-auto-cot.json \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --clustering_content=Q
#
#python $TASK_PATH \
#    --task=cweb-qa \
#    --input_gen_file=data/cweb-qa/$MODEL/zeroshot-cot-auto-cot.json \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --clustering_content=Q
#
#python $TASK_PATH \
#    --task=wikimh-qa \
#    --input_gen_file=data/wikimh-qa/$MODEL/zeroshot-cot-auto-cot.json \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --clustering_content=Q
#
#python $TASK_PATH \
#    --task=hotpot-qa \
#    --input_gen_file=data/hotpot-qa/$MODEL/zeroshot-cot-auto-cot.json \
#    --num_clusters=$NUM_CLUSTERS \
#    --method=$METHOD \
#    --clustering_content=Q

