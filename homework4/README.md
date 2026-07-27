# Homework 4 - A vision language model for tux

In this homework, we will train (fine-tune) two vision-language models on the SuperTuxKart data [here](https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip).
The first is a generative model, as known the Multimodal Large Language Model (MLLM); the second is a contrastive model, which is a simplified version of the Contrastive Language-Image Model (CLIP).
In the first part, we will focus on the most important aspect of the VLM pipeline: The data-pipeline.
We will use vision labels of the SuperTuxKart dataset to produce question/answer labels for the same set of images.
In the second part, we will focus on building a toy CLIP model and finetune it to do multi-choice question answering.
To fuel this, we will use the the SuperTuxKart dataset to generate some paired image-captions data.


The homework consists of two parts:

1. Train an VLM (`base_vlm.py`) using `finetune.py`.

2. Build a CLIP model (`clip.py`) based on the components in the VLM in Part 1.
Specifically, We use the vision model in the VLM as the CLIP's vision encoder, and use the LLM in the VLM as the CLIP's text encoder.
Finally, we train the CLIP to do multi-choice question answering.

The starter code contains a minimal training script and dataset.

- `data.py` load a dataset of *images*, *questions*, and *answers* specified in a json file. See `data/train_demo/balanced_qa_pairs.json` for an example.
- `base_vlm.py` sets of a VLM model that can both train and evaluate on the above training data
- `finetune.py` fine-tunes a VLM on a specific dataset
- `clip.py` trains a CLIP on a specific dataset

To get started, familiarize yourself with the starter code, download and unzip the data.

```bash
wget https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip -O supertux_data.zip
unzip supertux_data.zip
```

Then train a model on the demo data we provided

```bash
python -m homework.finetune demo_train
```

and benchmark this model

```bash
python -m homework.finetune test path/to/your/checkpoint
```

Do not expect the model to perform very well, after all it was trained on only 5 question-answer pairs.
Your task is to massively expand this training set.
The checkpoint path needs to include `adapter_config.json` and `adapter_model.safetensors`.

Build a similar data pipeline. Instead of (`question`， `answer`, `image_file`) triplet, CLIP takes as input (`caption`, `image_file`) pairs.

Implement the missing pieces, and you can train a CLIP model using

```bash
python -m homework.clip train
```

and test this model using

```bash
python -m homework.clip test path/to/your/checkpoint
```

The checkpoint path needs to include `adapter_config.json`, `adapter_model.safetensors`, and `additional_weights.pt`.

## Grading

Each part will take 50 pts.
To get 50pts on the first part, you should answer 70% of questions correctly.
The score falls off linearly till 0%.
To get 50pts on the second part, you should answer 70% of questions correctly.
The score falls off linearly till 20% (accuracy of random guess).
There is a 5pt extra credit for submissions reaching 85% accuracy (linearly from 80%).

## Building a VLM data-pipeline (50 pts)

Your main task is to massively expand the dataset for VLM training. Follow the 5 kind of questions given in our demo training set.

`generate_qa.py` provides some initial code to parse and visualize the supertux data.

Run

```bash
python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0
```

The visualize the extracted supertuxkart information and your generated questions.
Finally, write question-answer pairs into a `..._qa_pairs.json` file in `data/train/` and train your model using

```bash
python -m homework.finetune train
```

Do NOT train on the validation data provided.
It will inflate your validation accuracy, and unlikely generalizes to the test set.

## Train a CLIP model (50 pts)

Use the similar data-pipeline in the previous section to generate (`image_file`, `caption`) pairs.

Run

```bash
python -m homework.generate_captions check --info_file data/valid/00000_info.json --view_index 0
```

The visualize the extracted supertuxkart information and your generated captions.
Finally, write captions pairs into a `..._captions.json` file in `data/train/` and train your model using

```bash
python -m homework.clip train
```

## Submission

Once you finished the assignment, create a submission bundle using:

```bash
python3 bundle.py homework [YOUR UT ID]
```

Delete any old checkpoints from your homework directory to keep the model size below 50MB.

Submit the zip file on Canvas. Please note that the maximum file size our grader accepts is **50MB**. Please keep your solution compact.
Please double-check that your zip file was properly created, by grading it again:

```bash
python3 -m grader [YOUR UT ID].zip
```

## Online grader

We will use an automated grader through Canvas to grade all your submissions. There is a soft limit of **5** submissions per assignment. Please contact the course staff before going over this limit, otherwise your submission might be counted as invalid.

The online grading system will use a slightly modified version of Python and the grader:

- Please do not use the `exit` or `sys.exit` command, it will likely lead to a crash in the grader
- Please do not try to access, read, or write files outside the ones specified in the assignment. This again will lead to a crash. File writing is disabled.
- Network access is disabled. Please do not try to communicate with the outside world.
- Forking is not allowed!
- `print` or `sys.stdout.write` statements from your code are ignored and not returned.

Please do not try to break or hack the grader. Doing so will have negative consequences for your standing in this class and the program.

## Installation

We encourage using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to install the required packages.

```bash
conda create --name advances_in_deeplearning python=3.12 -y
conda activate advances_in_deeplearning
```

First, install [PyTorch](https://pytorch.org/get-started/locally/)

Then install additional dependencies:

```bash
pip install -r requirements.txt
```
