# Using Out-of-Domain Contrastive Instance Bundles to Improve Medical Question Answering

Code to train and analyze question answering language models on biomedical yes/no question answering.

To train models: python3 run.py --do_train --dataset *json dataset filepath or huggingface name* --model *filepath or huggingface name* --output_dir *filepath*

To evaluate models: python3 run.py --do_eval --dataset *json dataset filepath or huggingface name* --model *filepath or huggingface name* --output_dir *filepath*

Other notable arguments:
- datasubset (str): For huggingface datasets with different subsets
- max_length (int): Context window
- stride (int): Enables sliding context window with stride, for large training inputs
- bundle (bool): Enables contrastive estimation for training on instance bundles

To get prediction differences between two models: python3 find_discrepancies.py --file1 *error analysis filepath* --file2 *error analysis filepath* --output *filepath*

See paper.pdf for more detailed methodology
