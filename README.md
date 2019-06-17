<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## CVPR'19 Semi-Supervised Exploration for Image Retrieval
For GEM training, use the repo:
``` https://github.com/filipradenovic/cnnimageretrieval-pytorch```

For DIR latents, use pretrained weights and code from:
```https://github.com/figitaki/deep-retrieval ```

For GEM training params are:
``` '/media/himanshu/himanshu-dsk2/jeremy/copied_code/cnnimageretrieval-pytorch/export'
--gpu-id
0
--arch
resnet101
--pool
gem
--loss
contrastive
--loss-margin
.8
--optimizer
adam
--lr
2e-6
--neg-num
3
--query-size=2000
--pool-size=15000
--batch-size
10
--image-size
700
--test-whiten
retrieval-SfM-120k
--workers
10
--test-freq
2
--whitening
```

to create prebuild files for the following steps:

    `python create_prebuild.py`

to create the subgtaph:

    python write_same_label.py test_to_label.pkl label_to_index_sorted.pkl

this creates a file `label.txt`

to merge the two graphs:

    python  merge_graph.py stage2_gem_dir_concat_qe2_db2_trial1.bin label.txt


to run semisupervise egt, down the egt repo: https://github.com/layer6ai-labs/EGT and compile target


and run the following command:

    java -jar target/egt.jar -k 80 -q 117577 -t70 -p 100 -n 256016 --time merged_graph.txt test.txt


link for downloading the required files and weights:

`https://drive.google.com/drive/folders/1T6bkfRoV-d0sNbjaWqI9dFS3DT6HGour?usp=sharing`
