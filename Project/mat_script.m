for n=1:252
  filename = sprintf('data%d.mat',n) ;
  filepath = 'D:/nedc_tuh_eeg/edf2/eval/0';
  disp(filename);
  load(fullfile(filepath,filename));
  % S is a structure with the variables inside the mat file as its fields.
  % If you expect a variable called V, you can check this using ISFIELD
  savepath = 'edf/kurtograms3/eval/0';
  %savepath = 'edf';
  savename = sprintf('kurtograms%d',n) ;
  j = j+1;
  disp('Kurtogram start');
  tic;
  c = Fast_kurtogram(a,6,256,savepath,savename);
  toc;
  disp('Kurtogram end');
end