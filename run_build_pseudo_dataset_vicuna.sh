export PYTHONPATH=$PYTHONPATH:../
export CUDA_VISIBLE_DEVICES=4

python tasks/build_pseudo_dataset_vicuna.py \
    --data_dir=data/self-prompt-cot/vicuna-13b/ \
    --output_dir=data/self-prompt-cot \
    --demo_path=demos/multihop_demos.json \
    --yesno_demo_path=demos/yesno_demos.json \
    --overwrite_output \
