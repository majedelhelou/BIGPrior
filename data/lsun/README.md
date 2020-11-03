## LSUN data
The [lsun](https://www.yf.io/p/lsun) dataset consists of images from various indoor and outdoor scenes. 
To download the entire set for a category (both training and validation), the code provided by lsun can be used as follows.
```
cd code/
python download.py -c <category_name>
```
The validation sets can be downloaded individually from the [lsun webpage](http://dl.yf.io/lsun/scenes/)
The downloaded dataset is stored in lmdb format. To extract the images, the lsun code provides a script that can be used as follows.
```
python code/data.py export <lmdb_path> --out_dir <output directory>
```
Additional download options are available on their [github page](https://github.com/fyu/lsun). Note that the extracted images are in webp format which is not supported by mGAN and do not have the same sizes. To resize the images to 256 x 256 pixels and convert them to png format, we provide a script that can be used as follows.
```
cd code/
python webptopng.py -input_dir <input folder> -output_dir <output folder>
```

<!-- #### SUN
As well as the lsun validation datasets, we also conducted experiments on the images from the examples folder of the mganprior codes. As some of the images come from the SUN dataset, we also provide a way to download this dataset. The SUN images can be found at http://labelme.csail.mit.edu/Images/users/antonio/static_sun_database/. We also provide a script to scrape, download and resize all the images of a category using the `data/SUN/code/download.py` script, that can be used as:
```
cd ./data/SUN/code/
python download.py -url <image folder url> -output_dir <output folder>
```
 -->

**Credits to**: Martin Chatton