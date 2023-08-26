echo 'Training model and saving checkpoints'
python3 src/main.py

model 'Evaluating FID scores'
python3 src/fid.py
