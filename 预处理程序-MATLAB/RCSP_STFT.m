%一次性处理所有数据
clear,clc,close all;
tic;
for id1 = 1:1
    %次循环处理同一个被试的三个数据
    label = zeros(1,360);
    
    for id2 = 1:1
        %数据载入
        name = strcat('B0',num2str(id1),'0',num2str(id2),'T.gdf');
        disp(name);
        [s, h] = sload(name, 0, 'OVERFLOWDETECTION:OFF');
        
        %数据处理
        time1=120;%循环120次
        m1=500;%每次取出500个数据
        start_time=(h.TRIG+125);%读取出每次信号起始时间点.
        gnd = h.Classlabel(1:120)'; %gnd:the ground truth labels for the signal

        %生成RCSP的输入，经过STFT处理后的信号
        selCh = 500;  % selCh:number of selected channels/columns(最后选择的通道数量，不是数据拥有的通道数量！！！)
        numCls = 2;  % numCls:numblr of classes, 2 in this paper
        numTrn = 60;  % numTrn:number of training samples 
        width = 500;    %原始信号转置后的宽度，信号点数
        height = 3;    % 原始信号转置后的高度，也就是通道数
            
        raw_EEG = zeros(height,width,time1);%signal after STFT
        for i=1:time1
            i1=start_time(i);j1=i1+m1-1;%500暂时为信号前500个样本
            s_0=s(i1:j1,1:3); %s_0:raw signal of EEG for a single trial
            raw_EEG(:,:,i) = s_0'; 
        end
        
        gamma = [1e-15,1e-13,1e-11,1e-9,1e-7];
        signalRCSP = RCSP(selCh, numCls, raw_EEG, gnd, numTrn, gamma);
        %function allNewFtrs = STFT_RCSP(selCh,numCls,sa_STFT,gnd,numTrn)
        % selCh:number of selected channels/columns(最后选择的通道数量，不是数据拥有的通道数量！！！)
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
            
            %标签，生成csv文件
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
