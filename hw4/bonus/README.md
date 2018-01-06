## How to execute the program

### Step 1: Prepare training data
Type `bash ./download_dataset.sh ukiyoe2photo`

### Step 2: Replade target images with Anime images
`rm -R datasets/ukiyoe2photo/trainB` 
`rm -R datasets/ukiyoe2photo/testB`
`cp faces/* datasets/ukiyoe2photo/trainB` 
`cp faces/* datasets/ukiyoe2photo/tesetB` 

### Step 3: Training
`python3 main.py --dataset_dir=ukiyoe2photo`

### Step 4: Testing
1. Download model file from `https://gitlab.com/SimpleA/ADLxMLDS_2017/raw/master/bonus_model.zip`
2. Unzip the downloaded model file
3. Place the model file under the directory `checkpoint`
4. Type `python3 main.py --dataset_dir=ukiyoe2photo --phase=test --which_direction=BtoA`