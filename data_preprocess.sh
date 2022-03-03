nohup python tools/preprocess_data_cn.py \
            --input     /home/hustcsuser/YuanData/yuan/ \
            --vocab_path    /home/hustcsuser/lr/FinalProject \
            --output_path   /home/hustcsuser/YuanData/processed_yuan/ \
            --output_prefix   test_lr \
            --dataset_impl      mmap \
            --workers 8   \
            --sentence_splitter  True   \   
            >myfile.out 2>&1 &