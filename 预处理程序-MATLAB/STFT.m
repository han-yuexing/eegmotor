% clear,clc,close all;
% % [s, h] = sload('B0101T.gdf', 0, 'OVERFLOWDETECTION:OFF');
% [s, h] = sload('A01T.gdf', 0, 'OVERFLOWDETECTION:OFF');

function S_f=STFT_500_author2(forSTFT)
    
    fs=250; %����Ƶ��Ϊ250Hz
    N=length(forSTFT); %length of the signal x.����ȡ�ĸ�������ź�δ֪����ʱȡ��ǰ500��������
    n=0:N-1;
    t=n/fs;
    nfft=512;
        % ---------------------------------------------------------------------
        % ��ʱ����ҶSTFT�任
        % ---------------------------------------------------------------------
        
        win=window('hanning',64);%������Ϊ������
        [S1,F,T] = spectrogram(forSTFT,win,50,nfft,fs);
        %[S2,F,T] = spectrogram(cz,win,50,nfft,fs);
        %[S3,F,T] = spectrogram(c4,win,50,nfft,fs);

        % ---------------------------------------------------------------------
        % ������ȡ��mu(6-13Hz);beta(17-30Hz)
        % ---------------------------------------------------------------------
        f1=find(F>=6,1);
        f2=find(F>13,1)+1;
        f3=find(F>=17,1);
        f4=find(F>30,1)+1;
        
        Sa1 = abs(S1(f1:f2,:));
        mSa1 = mean(Sa1(:));
        %disp(size(Sa1));
        
        Sa2 = abs(S1(f3:f4,:));
        mSa2 = mean(Sa2(:));
        %disp(size(Sa2));
        %Sa1 = [abs(S1(f1:f2,:));abs(S3(f1:f2,:))];
        %Sa2 = [abs(S2(f3:f4,:));abs(S3(f3:f4,:))];
        %mSa2 = mean(Sa2(:));
        %mSz1 = mean(mean(abs(S2(f1:f2,:))));
        %mSz2 = mean(mean(abs(S2(f3:f4,:))));
        
        image_mu = Sa1/mSa1;
        %imshow(image_mu);
        image_beta = Sa2/mSa2;
        imshow(image_beta)
        
        S_f = [image_mu;imresize(image_beta, [16 32])];
        
        image_beta2 = imresize(image_beta, [16 32]);
        
end