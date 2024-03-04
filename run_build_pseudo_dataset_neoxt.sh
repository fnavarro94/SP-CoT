export PYTHONPATH=$PYTHONPATH:../
export CUDA_VISIBLE_DEVICES=2,3

python tasks/build_pseudo_dataset_neoxt.py \
    --data_dir=data/self-prompt-cot/gpt-neoxt-chat/ \
    --output_dir=data/self-prompt-cot \
    --demo_path=demos/multihop_demos.json \
    --yesno_demo_path=demos/yesno_demos.json \
    --overwrite_output
