export XDG_CACHE_HOME=cache/

# Colorization
python code/mganprior/colorization.py --gan_model pggan_bedroom --target_images data/lsun/data/bedroom --outputs inter_data/col_bedroom --composing_layer 6 --z_number 20
python code/mganprior/colorization.py --gan_model pggan_churchoutdoor --target_images data/lsun/data/church --outputs inter_data/col_church --composing_layer 6 --z_number 20

# Inpainting: crop center 64x64
python code/mganprior/inpainting.py --gan_model pggan_churchoutdoor --target_images data/lsun/data/church --outputs inter_data/inpcrop_church --mask mask_center.png --composing_layer 4 --z_number 30
python code/mganprior/inpainting.py --gan_model pggan_bedroom --target_images data/lsun/data/bedroom --outputs inter_data/inpcrop_bedroom --mask mask_center.png --composing_layer 4 --z_number 30
python code/mganprior/inpainting.py --gan_model pggan_conferenceroom --target_images data/lsun/data/conference --outputs inter_data/inpcrop_conference --mask mask_center.png --composing_layer 4 --z_number 30

# Inpainting: variable mask
python code/mganprior/inpainting.py --gan_model pggan_churchoutdoor --target_images data/lsun/data/church --outputs inter_data/inpvar_church --varmask True --composing_layer 4 --z_number 30
python code/mganprior/inpainting.py --gan_model pggan_bedroom --target_images data/lsun/data/bedroom --outputs inter_data/inpvar_bedroom --varmask True --composing_layer 4 --z_number 30
python code/mganprior/inpainting.py --gan_model pggan_conferenceroom --target_images data/lsun/data/conference --outputs inter_data/inpvar_conference --varmask True --composing_layer 4 --z_number 30

# Denoising: blind AWGN (5-50)
python code/mganprior/denoising.py --gan_model pggan_churchoutdoor --target_images data/lsun/data/church --outputs inter_data/denblind_church --sigma_min 5 --sigma_max 50 --composing_layer 4 --z_number 30
python code/mganprior/denoising.py --gan_model pggan_bedroom --target_images data/lsun/data/bedroom --outputs inter_data/denblind_bedroom --sigma_min 5 --sigma_max 50 --composing_layer 4 --z_number 30
python code/mganprior/denoising.py --gan_model pggan_conferenceroom --target_images data/lsun/data/conference --outputs inter_data/denblind_conference --sigma_min 5 --sigma_max 50 --composing_layer 4 --z_number 30