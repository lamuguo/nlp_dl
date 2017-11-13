
# generate data for pairs of text

python2 test_preparation_for_classify.py

cd ../textmatch

# train the model
python2 main.py --phase train --model_file models/arci_classify.config


# predict with the model

python2 main.py --phase predict --model_file models/arci_classify.config
