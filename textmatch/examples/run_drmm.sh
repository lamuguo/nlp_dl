
# generate match histogram
python2 test_preparation_for_ranking.py
python2 test_histogram_generator.py 

cd ../textmatch

# configure the model file
#cd models

# train the model
python2 main.py --phase train --model_file models/drmm.config

# test the model
python2 main.py --phase predict --model_file models/drmm.config
