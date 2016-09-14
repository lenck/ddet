function downloaded = provision( url_file, tgt_dir)
% PROVISION Provision a binary file from an archive
%   PROVISION(URL_FILE, TGT_DIR) Downloads and unpacks the archive from
%   URL_FILE to TGT_DIR folder, if not already done.
%
%   Uses an empty file:
%      TGT_DIR/.URL_FILE_NAME.done
%   as an indicator that the folder had been already provisioned.

% Copyright (C) 2016 Karel Lenc.
% All rights reserved.
%
% Tishis file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
downloaded = false;
if ~exist(url_file, 'file')
  error('Unable to find the URL file %s.', url_file);
end;
[~, url_file_nm] = fileparts(url_file);
done_file = fullfile(tgt_dir, ['.', url_file_nm, '.done']);
if exist(done_file, 'file'), return; end;
[~,~] = mkdir(tgt_dir);
url = utls.readfile(url_file);
for ui = 1:numel(url)
  p_unpack(url{ui}, tgt_dir);
end
downloaded = true;
f = fopen(done_file, 'w'); fclose(f);
end

function p_unpack(url, tgt_dir)
[~, wget_p] = system('which wget');
if exist(strtrim(wget_p), 'file');
  [~, fname, ext] = fileparts(url);
  tar_file = fullfile(tgt_dir, [fname, ext]);
  if ~exist(tar_file, 'file')
    fprintf(isdeployed+1, 'Downloading %s -> %s.\n', url, tar_file);
    ret = system(sprintf('wget %s -O %s', url, tar_file));
    if ret ~= 0
      fprintf(isdeployed+1, 'wget failed.');
      delete(tar_file);
    end;
  end
  if exist(tar_file, 'file')
    fprintf(isdeployed+1, 'Unpacking %s -> %s. This may take a while...\n', ...
      tar_file, tgt_dir);
    m_unpack(tar_file, tgt_dir);
  else
    m_unpack(url, tgt_dir);
  end
else
  m_unpack(url, tgt_dir);
end
end

function m_unpack(url, tgt_dir)
fprintf(isdeployed+1, ...
  'Downloading %s -> %s using MATLAB, this may take a while...\n',...
  url, tgt_dir);
[~, ~, ext] = fileparts(url);
switch ext
  case '.gz'
    untar(url, tgt_dir);
  case '.zip'
    unzip(url, tgt_dir);
end
end