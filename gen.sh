echo 'Training model and saving checkpoints'
python3 main.py

model 'Evaluating FID scores'
python3 fid.py
