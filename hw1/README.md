## Testing
- To run the RNN model, type `./hw1_rnn.sh $(path_to_data) $(path_of_output)`
- To run the CNN model, type `./hw1_cnn.sh $(path_to_data) $(path_of_output)`
- To run the bset model, type `./hw1_best.sh $(path_to_data) $(path_of_output)`. The best model is the RNN model, so the `hw1_best.sh` is identical to `hw1_rnn.sh`

## Training
- To train the RNN model, type `python model_rnn.py $(path_to_data)`. Here `python` is assumed to be an alias of `python3`. The model will be saved to `./rnn_model`.
- To train the CNN model, type `python model_cnn.py $(path_to_data)`. Here `python` is assumed to be an alias of `python3`. The model will be saved to `./cnn_model`.
