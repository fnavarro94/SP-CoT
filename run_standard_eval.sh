export PYTHONPATH=$PYTHONPATH:../
export METHOD=zeroshot-dpr-retrieval
export TASK=tasks/standard_evaluation.py
export MODEL=vicuna-13b
export FILE_NAME=$METHOD.json


export DATASET=musique-qa
python $TASK \
    --input_file=data/$DATASET/$MODEL/$FILE_NAME \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --extract_answer



export DATASET=hotpot-qa
python $TASK \
    --input_file=data/$DATASET/$MODEL/$FILE_NAME \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --extract_answer


export DATASET=wikimh-qa
python $TASK \
    --input_file=data/$DATASET/$MODEL/$FILE_NAME \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --extract_answer


export DATASET=cweb-qa
python $TASK \
    --input_file=data/$DATASET/$MODEL/$FILE_NAME \
    --task=$DATASET \
    --method=$METHOD \
    --model_name=$MODEL \
    --extract_answer

