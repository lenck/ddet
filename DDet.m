classdef DDet < handle
%DDET Implementation of the Convariant feature detector.
%  DDet implements the local feature detection using the network trained on
%  regressing relative translation between two patches. Accummulates the
%  relative transformations using biliniear voting.

% Copyright (C) 2016 Karel Lenc.
% All rights reserved.
%
% Tishis file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
  
  properties (SetAccess=public, GetAccess=public)
    Opts = struct('thr',3, 'defscale', 3);
    Net
    Args;
  end
  
  methods
    function obj = DDet(net, varargin)
      %DDET Construct DDet object
      %  obj = DDET(NET) construc a DDET using the network NET. Network
      %  must be a `dagnn.DagNN` object and must have an input `x0a` and
      %  output `feata`.
      %
      %  Accepts the following options:
      %
      %  `thr` :: 3
      %    Detection threshold. Number of regressed locations which must have
      %    voted to the particular location in order to be considered as a valid
      %    detection.
      assert(isa(net, 'dagnn.DagNN'), 'Invalid network');
      assert(ismember('x0a', net.getInputs()), 'Input x0a not found.');
      assert(ismember('x0a', net.getInputs()), 'Input x0a not found.');
      assert(ismember('feata', net.getOutputs()), 'Output feata not found.');
      obj.Net = net;
      [obj.Opts, obj.Args] = vl_argparse(obj.Opts, varargin);
    end
    
    function [frames, desc, info] = detect(obj, im)
      %DETECT Detect local features in image im.
      %  FMS = obj.detect(im) Detect features in image im.
      %  [FMS, ~, INFO] obj.detect(im) Additionally returns an info
      %  structure with fields:
      %
      %  `im_accum`   - The accummulated votes.
      %  `vfield`     - Regressed vector field.
      %  `peakScores` - Score for each detected feature.
      
      desc = [];
      [pts_im, geom, offs_] = obj.eval(obj.Net, im, obj.Args{:});
      
      % Accummulate the votes using bilinear function
      conf = double(pts_im(3, :));
      im_sz = [size(im, 1), size(im, 2)];
      im_accum = double(zeros(im_sz));
      pts_im = double(pts_im(1:2, :));
      pts_x = floor(pts_im(1,:)); pts_y = floor(pts_im(2,:));
      x_a = pts_im(1,:) - pts_x; x_b = 1 - x_a;
      y_a = pts_im(2,:) - pts_y; y_b = 1 - y_a;
      im_accum = vl_binsum(im_accum, conf .* x_b .* y_b, sub2ind(im_sz, pts_y, pts_x));
      im_accum = vl_binsum(im_accum, conf .* x_a .* y_b, sub2ind(im_sz, pts_y, pts_x+1));
      im_accum = vl_binsum(im_accum, conf .* x_a .* y_a, sub2ind(im_sz, pts_y+1, pts_x+1));
      im_accum = vl_binsum(im_accum, conf .* x_b .* y_a, sub2ind(im_sz, pts_y+1, pts_x));
      
      pts = imregionalmax(im_accum);
      [pts_y, pts_x] = find(pts);
      pts_value = im_accum(pts);
      pts = [pts_x(:) pts_y(:)]';
      
      frames_sel = pts_value > obj.Opts.thr;
      frames = pts(:, frames_sel);
      
      info = struct('im_accum', im_accum);
      info.vfield = zeros(size(im, 1), size(im, 2), size(offs_, 3));
      info.vfield(floor(geom.y_offs), floor(geom.x_offs), :) = offs_ .* geom.hsz(1);
      info.peakScores = pts_value(frames_sel)';
      
      frames = [frames; obj.Opts.defscale * ones(1, size(frames, 2))];
    end
  end
  
  methods (Static)
    function [ pts, geom, offs ] = eval( evnet, im, varargin )
      opts.featureLayer = 'x0a';
      opts.outLayer = 'feata';
      opts.gpu = [];
      opts = vl_argparse(opts, varargin);
      
      assert(isfield(evnet.meta, 'data_mean'), 'Missing net.meta.data_mean');
      evnet.vars(evnet.getVarIndex(opts.featureLayer)).precious = true;
      
      % Image preprocessing
      if size(im, 3) == 3, im = rgb2gray(im); end
      if isa(im, 'uint8'), im = im2single(im); end
      
      m = evnet.meta.data_mean;
      im_n = bsxfun(@minus, im, m);
      
      geom.hsz = (evnet.meta.inputSize(1:2)./2)';
      rfs = evnet.getVarReceptiveFields(opts.featureLayer);
      outsz = evnet.getVarSizes({opts.featureLayer, size(im)});
      outlayerIdx = evnet.getVarIndex(opts.outLayer);
      
      %! In this case the x_offs is the centre of the receptive field
      geom.x_offs = ((1:outsz{outlayerIdx}(2)) - 1) * ...
        rfs(outlayerIdx).stride(1) + geom.hsz(2) + 0.5;
      geom.y_offs = ((1:outsz{outlayerIdx}(1)) - 1) * ...
        rfs(outlayerIdx).stride(2) + geom.hsz(1) + 0.5;
      
      [ys, xs] = ndgrid(geom.y_offs, geom.x_offs);
      geom.tl_anchors = [xs(:)'; ys(:)'];
      
      if strcmp(evnet.device, 'gpu'), im_n = gpuArray(im_n); end;
      
      evnet.eval({opts.featureLayer, single(im_n)});
      pts = gather(evnet.vars(evnet.getVarIndex(opts.outLayer)).value);
      pts_size = size(pts);
      pts = reshape(pts, [], 2)';
      offs = reshape(pts', pts_size);
      
      pts = [pts; ones(1, size(pts, 2))];
      pts(1:2,:) = bsxfun(@times, pts(1:2,:), geom.hsz);
      pts(1:2,:) = pts(1:2,:) + geom.tl_anchors;
      % Remove points reaching out of image
      pts(:, pts(1, :) < 1 | pts(2, :) < 1 | ...
        pts(1,:) >= size(im, 2) | pts(2,:) >= size(im, 1)) = [];
    end
    
  end
end
