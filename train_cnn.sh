# Colorization
python code/train.py --experiment col_bedroom --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5
python code/train.py --experiment col_church --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5


# Blind denoising
python code/train.py --experiment denblind_bedroom --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5
python code/train.py --experiment denblind_church --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5
python code/train.py --experiment denblind_conference --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5


# Inpainting CROP (64x64 center crop)
python code/train.py --experiment inpcrop_bedroom --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5
python code/train.py --experiment inpcrop_church --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5
python code/train.py --experiment inpcrop_conference --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5


# Inpainting VAR (variable randomized mask)
python code/train.py --experiment inpvar_bedroom --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5
python code/train.py --experiment inpvar_church --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5
python code/train.py --experiment inpvar_conference --lr 0.01 --batch_size 8 --backbone D --phi_weight 1e-5