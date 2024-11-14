export PYTHONPATH=$PYTHONPATH:../

python tasks/build_pseudo_dataset.py \
    --data_dir=data/self-prompt-cot/gpt-4o-mini \
    --output_dir=data/self-prompt-cot \
    --demo_path=demos/multihop_demos.json \
    --do_composition \
    --do_completion \
    --do_filtering \
    --do_merge \
    --yesno_demo_path=demos/yesno_demos.json 
    
