% Illustrates usage of 'Fast_kurtogram' on a synthetic signal
% -----------------------------------------------------------
% This example illustrates the detection of weak repetitive transients
% hidden in stationary noise; their theoretical spectral content is in band [.15 .19].
%
% --------------------------
% Author: J. Antoni
% Last Revision: 12-2014
% --------------------------
% created my ANSHUL
clear
close all;
clc
%% load springers data and options
load('Springer_B_matrix.mat');
load('Springer_pi_vector.mat');
load('Springer_total_obs_distribution.mat');
springer_options   = default_Springer_HSMM_options;

%% load data
%% Load data and resample data
names=['a','b','c','d','e','f'];
for i=2:6
    data_dir = [pwd filesep 'Physionet2016' filesep ['training-',names(i)] filesep];
    referenceFile=[data_dir 'REFERENCE.csv'];
    [lavel,records,~] = xlsread(referenceFile);
    features=zeros(length(records),43);
    for j=1:length(records)
        names(i);
        fname=records{j};
        recordName = [data_dir fname];
        [PCG, SamplingFs] = audioread([recordName '.wav']); 

        if length(PCG)>20*SamplingFs
            PCG = PCG(1:20*SamplingFs);
        end
        
        newFs=1000;
        PCG_resampled = resample(PCG,newFs,SamplingFs);
%%  kurtogram
        flag=lavel(j);
        if(flag==1)
            fig=[pwd filesep 'Kurtogram_Images_data' filesep 'Abnormal' filesep fname '.jpg'];
        else
            fig=[pwd filesep 'Kurtogram_Images_data' filesep 'Normal' filesep fname '.jpg'];
        end
        x=PCG_resampled;
        Fs = newFs;         % sampling frequency

        %figure,plot(x,'k'),title('Signal with hidden repetitive transients')

        nlevel = 6;     % number of decomposition levels
       %figure, plot(PCG_resampled)
       % Fast Kurtogram
        added_path = [pwd,'/KurtogramV4'];
        addpath(added_path);
        
        Fast_kurtogram_updated(x,nlevel,Fs,fig);
        
        rmpath(added_path);
        
    end
end

