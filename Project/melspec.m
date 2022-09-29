for n=1:300
  filename = sprintf('data%d.mat',n) ;
  filepath = 'D:/nedc_tuh_eeg/edf2/eval/1';
  disp(filename);
  load(fullfile(filepath,filename));
  % S is a structure with the variables inside the mat file as its fields.
  % If you expect a variable called V, you can check this using ISFIELD
  savepath = 'edf/melSpectrogram/eval/1';
  %savepath = 'edf';
  savename = sprintf('melSpectrograms%d',n) ;
  j = j+1;
  a = a.';
  disp('melSpectrogram start');
  tic;
  melSpectrogram(a,6400);
  
  hFig = gcf;
  hAx  = gca;
  % set the figure to full screen
  %set(hFig,'units','normalized','outerposition',[0 0 1 1]);
  set(hFig,'position',[0 0 256 256]);
  % set the axes to full screen
  %set(hAx,'Unit','normalized','Position',[0 0 1 1]);
  set(hAx,'Unit','normalized','Position',[0 0 1 1]);
  % hide the toolbar
  set(hFig,'menubar','none')
  % to hide the title
  set(hFig,'NumberTitle','off');
  saveas(gcf,fullfile(savepath,savename), 'png')
  close(gcf)
  toc;
  disp('melSpectrogram end');
end