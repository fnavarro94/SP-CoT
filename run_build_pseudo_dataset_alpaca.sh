export PYTHONPATH=$PYTHONPATH:../
export CUDA_VISIBLE_DEVICES=7

python tasks/build_pseudo_dataset_alpaca.py \
    --data_dir=data/self-prompt-cot/alpaca-13b/ \
    --output_dir=data/self-prompt-cot \
    --demo_path=demos/multihop_demos.json \
    --yesno_demo_path=demos/yesno_demos.json \
    --overwrite_output
