function setup()
% SETUP Setup the environment

% Copyright (C) 2016 Karel Lenc.
% All rights reserved.
%
% Tishis file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% Setup VLFeat, if not in path
if ~exist('vl_covdet', 'file')
  utls.provision('vlfeat.url', 'vlfeat');
  run(fullfile(getlatest('vlfeat', 'vlfeat'), 'toolbox', 'vl_setup.m'));
end

% Setup MatConvNet, if not in path
if ~exist('vl_nnconv', 'file')
  utls.provision('matconvnet.url', 'matconvnet');
  run(fullfile(getlatest('matconvnet', 'matconvnet'), 'matlab', 'vl_setupnn.m'));
  
  if isempty(strfind(which('vl_nnconv'), mexext))
    fprintf('MatConvNet not compiled. Attempting to run `vl_compilenn` (CPU ONLY!).\n');
    fprintf('To compile with a GPU support, see `help vl_compilenn`.');
    vl_compilenn('EnableImreadJpeg', false);
  end
end

utls.provision(fullfile('nets', 'nets.url'), 'nets');
end

function out = getlatest(path, name)
sel_dir = dir(fullfile(path, [name '*']));
sel_dir = sel_dir([sel_dir.isdir]);
sel_dir = sort({sel_dir.name});
out = fullfile(path, sel_dir{end});
end