## Code usage
### Testing
./hw2_seq2sreq.sh $(data_directory) $(testing data output file) $(peer review output file)

The testing code will first download my model from GitLab, and the time for download may exceed 10 minutes, depending on the network speed.
After download, the script will automatically perform prediction on the testing data and peer review data

### Training
python model_seq2seq.py $(data_directory)

### Packages used by this program
1. Tensorflow 1.3
2. Python 3.4
3. Numpy 1.13
4. Pandas 0.20
 
