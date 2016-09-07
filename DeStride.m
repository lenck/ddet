classdef DeStride < dagnn.Filter
% DESTRIDE A baisc implementation of a-trous algorithm
%   DESTRIDE simply allows to evaluate a network with an operation with
%   stride >1 densely. It achieves it by sampling the data with a different
%   offset, computing the operation, and joining the data back. In this way
%   it achieves to have the same size of the input and output.
  
% Copyright (C) 2016 Karel Lenc.
% All rights reserved.
%
% Tishis file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
  
  properties
    destride;
    layer;
  end
  
  properties (Hidden, Transient)
    offsets;
  end

  methods
    function outputs = forward(obj, inputs, params)
      out = [];
      assert(numel(inputs) == 1);
      strd = obj.destride;
      for oi = 1:size(obj.offsets, 2)
        o = obj.offsets(:, oi);
        nin = {inputs{1}(o(2):strd(2):end, o(1):strd(1):end, :, :)};
        out_r = obj.layer.forward(nin, params);
        out_r = out_r{1};
        if isempty(out)
          out = zeros(size(out_r, 1)*obj.destride(2), ...
            size(out_r, 2)*obj.destride(1), size(out_r, 3), ...
            size(out_r, 4), 'like', out_r);
        end
        % Pick the indexes based on the output size -> zero padding if
        % needed
        oidx_y = ((1:size(out_r, 1)) - 1)*strd(2) + o(2);
        oidx_x = ((1:size(out_r, 2)) - 1)*strd(1) + o(1);
        out(oidx_y, oidx_x, :, :) = out_r;
      end
      outputs = {out};
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.layer.getKernelSize() ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      strd = obj.destride;
      inputSizes{1}(1) = numel(1:strd(2):inputSizes{1}(1));
      inputSizes{1}(2) = numel(1:strd(1):inputSizes{1}(2));
      outputSizes = obj.layer.getOutputSizes(inputSizes);
      outputSizes{1}(1:2) = outputSizes{1}(1:2) .* strd;
      assert(numel(outputSizes) == 1);
    end
    
    function set.destride(obj, destride)
      obj.destride = destride;
      [ofy, ofx] = ndgrid(1:destride(1), 1:destride(2));
      obj.offsets = [ofy(:)'; ofx(:)'];
    end
    
    function rfs = getReceptiveFields(obj)
      ks = obj.getKernelSize() ;
      y1 = 1 - obj.pad(1) ;
      y2 = 1 - obj.pad(1) + ks(1)*obj.destride(1) - 1 ;
      x1 = 1 - obj.pad(3) ;
      x2 = 1 - obj.pad(3) + ks(2)*obj.destride(2) - 1 ;
      h = y2 - y1 + 1 ;
      w = x2 - x1 + 1 ;
      rfs.size = [h, w] ;
      rfs.stride = obj.stride ;
      rfs.offset = [y1+y2, x1+x2]/2 ;
    end
  end
end
