
%out_folder = '../results/sr_real_pggan_celebahq';
filepaths = dir(fullfile(out_folder, '*.png'));
A = zeros(length(filepaths),1);

for idx_im = 1:length(filepaths)
    im_name = filepaths(idx_im).name;
    img = imread(fullfile(out_folder, im_name));
    niqeI = niqe(img);
    A(idx_im) = niqeI;
    fprintf('NIQE score for image: %s, %0.4f.\n',im_name,niqeI);
end
fprintf('mean NIQE score mean: %0.4f.\n',mean(A));
fprintf('NIQE score std: %0.4f.\n',std(A));
exit
