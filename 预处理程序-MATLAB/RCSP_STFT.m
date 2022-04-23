%һ���Դ�����������
clear,clc,close all;
tic;
for id1 = 1:1
    %��ѭ������ͬһ�����Ե���������
    label = zeros(1,360);
    
    for id2 = 1:1
        %��������
        name = strcat('B0',num2str(id1),'0',num2str(id2),'T.gdf');
        disp(name);
        [s, h] = sload(name, 0, 'OVERFLOWDETECTION:OFF');
        
        %���ݴ���
        time1=120;%ѭ��120��
        m1=500;%ÿ��ȡ��500������
        start_time=(h.TRIG+125);%��ȡ��ÿ���ź���ʼʱ���.
        gnd = h.Classlabel(1:120)'; %gnd:the ground truth labels for the signal

        %����RCSP�����룬����STFT�������ź�
        selCh = 500;  % selCh:number of selected channels/columns(���ѡ���ͨ����������������ӵ�е�ͨ������������)
        numCls = 2;  % numCls:numblr of classes, 2 in this paper
        numTrn = 60;  % numTrn:number of training samples 
        width = 500;    %ԭʼ�ź�ת�ú�Ŀ�ȣ��źŵ���
        height = 3;    % ԭʼ�ź�ת�ú�ĸ߶ȣ�Ҳ����ͨ����
            
        raw_EEG = zeros(height,width,time1);%signal after STFT
        for i=1:time1
            i1=start_time(i);j1=i1+m1-1;%500��ʱΪ�ź�ǰ500������
            s_0=s(i1:j1,1:3); %s_0:raw signal of EEG for a single trial
            raw_EEG(:,:,i) = s_0'; 
        end
        
        gamma = [1e-15,1e-13,1e-11,1e-9,1e-7];
        signalRCSP = RCSP(selCh, numCls, raw_EEG, gnd, numTrn, gamma);
        %function allNewFtrs = STFT_RCSP(selCh,numCls,sa_STFT,gnd,numTrn)
        % selCh:number of selected channels/columns(���ѡ���ͨ����������������ӵ�е�ͨ������������)
        % numCls:numblr of classes, 2 in this paper
        % sa_STFT:signal after STFT
        %gnd:the ground truth labels for the signal
        % numTrn:number of training samples 
        finalData = permute(signalRCSP,[1,3,2]);
        [numFtrs, numParas, numSam] = size(finalData);
        for id3 = 1:numParas
            
            for id4 = 1:numSam
                forSTFT = finalData(:,id3,id4);
                Train = STFT(forSTFT);
                %a_Train = abs(Train);
                %m_Train = mean(mean(a_Train));
                %Train2 = a_Train/m_Train;
                Train2 = mapminmax(Train,0,1);
                numFilename = num2str((id4)+120*(id2-1));
                %filename0=['E:\BCI Pre-Processing\Data_rcsp_stft\BC4_2b_F2\Train\s' num2str(id1) '\p' num2str(id3) '\' num2str(numFilename) '.csv'];
                %dlmwrite(filename0,Train2,'precision', '%.4f')
            end
            
            %��ǩ������csv�ļ�
            startLabel = 120*(id2-1)+1;
            endLabel = 120*id2;
            label(:,startLabel:endLabel) = gnd;
            %filename1=['E:\BCI Pre-Processing\Data_rcsp_stft\BC4_2b_F2\Label\s' num2str(id1) '\' 'label.csv'];
            %dlmwrite(filename1,label);
            
        end

            
 
     end            

    
end

toc;
fprintf('================= calculation has been finished. =================')
