function allNewFtrs = STFT_RCSP2(selCh,numCls,raw_EEG,gnd,numTrn,gamma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% selCh,number of selected channels/columns
% numCls, numblr of classes, 2 in this paper
% sa_STFT, signal after STFT
%gnd:数据标签
% numTrn, number of training samples

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% selCh=32;%number of selected channels/columns
% numCls=2;%two classes only
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%Regularization parameter used in the paper
betas=1e-2;
gammas=gamma;
numBeta=length(betas);
numGamma=length(gammas);
numRegs=numBeta*numGamma;
regParas=zeros(numRegs,2);
iReg=0;
for ibeta=1:numBeta
    beta=betas(ibeta);
    for igamma=1:numGamma
        gamma=gammas(igamma);
        iReg=iReg+1;
        regParas(iReg,1)=beta;
        regParas(iReg,2)=gamma;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Calculate the generic covariance matrices from generic training trials for
%two classes, equation (6) in the paper
%[genSs,genMs]=genericCov();%implement, please refer to line 65-79 of RegCsp.m

[~,numCh,~] = size(raw_EEG);%Input data, see documentation
genSs = zeros(numCh,numCh,2);
genMs = [0;0];
Ns=zeros(2,1);
for i=1:2%for each class
    Idxs=find(gnd==i);
    EEG=raw_EEG(:,:,Idxs);
    Ns(i)=length(Idxs);
    C=zeros(numCh,numCh,Ns(i));%Sample covariance matrix, equation (1) in the paper
    for trial=1:Ns(i)
        E=EEG(:,:,trial)'; 
        tmpC = (E*E');        
        C(:,:,trial) = tmpC./trace(tmpC);%normalization
    end 
    Csum=sum(C,3);
    genSs(:,:,i) = Csum;
    %Calculate the parameter of genSs, Sc^ in the equation(4) in the paper
    
    %Calculate the parameter of genMs, M^ in the equation(4) in the paper
    genMs(i) = Ns(i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Load data
SubDir=[];%please specify the directory of EEG data (plus groundtruth) for a subject here
%load([EEGdatafilename]);%please specify EEGdatafilename, it should contain "EEG" for the data and "gnd" for the ground truth
% numTrn=60; %please specify number of training samples
[numT,numCh,nTrl] = size(raw_EEG);%suppose data is loaded to variable "EEG"
trainIdx=1:numTrn;%take the first numTrn samples for training
testIdx=(numTrn+1):length(gnd);%the rest for testing
fea2D_Train = raw_EEG(:,:,trainIdx);
gnd_Train = gnd(trainIdx);
fea2D = raw_EEG;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%Multiple R-CSPs%%%%%%%%%%%
allNewFtrs=zeros(selCh,nTrl,numRegs);%Projected Features for R-CSP
for iReg=1:numRegs
    regPara=regParas(iReg,:);
    %=================RegCSP==========================%     
    %Prototype: W=RegCsp(EEGdata,gnd,genSs,genMs,beta,gamma)
    prjW= RegCsp2(fea2D_Train,gnd_Train,genSs,genMs,regPara(1),regPara(2));
    %=================Finished Training==========================
    prjW=prjW(1:selCh,:);%Select columns 
    newfea=zeros(selCh,nTrl);
    %newfea=zeros(1,nTrl);  %我改的
    for iCh=1:selCh
        for iTr=1:nTrl
            infea=fea2D(:,:,iTr)';
            prjfea=prjW(iCh,:)*infea;
            %disp(size(prjfea));
            newfea(iCh,iTr)=log(var(prjfea));
        end
    end
    allNewFtrs(:,:,iReg)=newfea;
end
end
