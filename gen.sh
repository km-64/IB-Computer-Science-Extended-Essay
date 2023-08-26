echo 'Training model and saving checkpoints'
python3 src/main.py

echo 'Evaluating FID scores'
python3 src/fid.py
