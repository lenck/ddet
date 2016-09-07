%% Example use of the DDet
setup();

%% Detect the features

net_name = 'detnet_s2.mat';
net = dagnn.DagNN.loadobj(load(fullfile('nets', net_name)));

% Uncomment the following lines to compute on a GPU
% (works only if MatConvNet compiled with GPU support)
% gpuDevice(1);  net.move('gpu');

detector = DDet(net, 'thr', 4);

im = vl_impattern('box');
[frames, ~, info] = detector.detect(im);

%% Plot the results

figure(1); clf;
subplot(2,2,1);
imshow(repmat(im, 1, 1, 3));
title('Original image');

subplot(2,2,2);
imshow(repmat(im, 1, 1, 3));
hold on; scatter(frames(1, :), frames(2, :), ...
  info.peakScores, info.peakScores, 'filled');
colormap jet;
title('Detecions');
text(0, size(im, 1)+10, 'Area represents feature strength.');

subplot(2,2,3);
accum = info.im_accum ./ max(info.im_accum(:)); alpha = 0.15;
imshow(cat(3, sqrt(accum)*(1-alpha) + im*alpha, im*alpha, im*alpha));
axis image;
title('Accummulated locations');
text(0, size(im, 1)+10, 'SQRT for better visibility');

subplot(2,2,4);
imshow(repmat(im, 1, 1, 3)); hold on;
quiver(info.vfield(:,:,1), info.vfield(:,:,2), 0);
title('Vector field of the regressed locations');

if ~exist('example.png', 'file')
  vl_printsize(1);
  print('-dpng', '-r100', 'example.png');
end