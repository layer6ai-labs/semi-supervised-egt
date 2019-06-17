<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## CVPR'19 Semi-Supervised Exploration for Image Retrieval
to create the subgtaph:
    `python write_same_label.py test_to_label.pkl label_to_index_sorted.pkl`

this creates a file `label.txt`

to merge the two graphs:
    `python  merge_graph.py stage2_gem_dir_concat_qe2_db2_trial1.bin label.txt`


to run semisupervise egt, down the  egt repo and run the following command:
    `java -jar target/egt.jar -k 80 -q 117577 -t70 -p 100 -n 256016 --time merged_graph.txt test.txt`
