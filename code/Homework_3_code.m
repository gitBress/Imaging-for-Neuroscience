% IMAGING FOR NEUROSCIENCE
 
% Final project, Homework 3
% AUTHOR: Giulia Bressan
% I.D.: 1206752

%% RESTING STATE fMRI ANALYSIS

clear all
close all
clc
%% 1. DATA PREPROCESSING

% Load data
grey_matter = load_untouch_nii('..\Saved_Data\GM.nii');
GM = double(grey_matter.img)/255;
white_matter = load_untouch_nii('..\Saved_Data\WM.nii');
WM = double(white_matter.img)/255;
cerebro_spinal_fluid = load_untouch_nii('..\Saved_Data\CSF.nii');
CSF = double(cerebro_spinal_fluid.img)/255;

fMRI = load_untouch_nii('..\Data\FMRI\FMRI_2_T1_2mm.nii');
data = double(fMRI.img);
size_data = size(data);

clear grey_matter
clear white_matter
clear cerebro_spinal_fluid

%% 1.b Create masks

% Thresholding 
maskGM = GM > 0.9;
maskWM = WM > 0.9;
maskCSF = CSF > 0.8;

% Mask erosion
se = [1;1];
maskGM_e = imerode(maskGM,se);
maskWM_e = imerode(maskWM,se);
maskCSF_e = imerode(maskCSF,se);

ex_slice = 45;
figure,
% for i=1:90
% ex_slice = i;
    subplot(231)
    imagesc(squeeze(GM(:,ex_slice,:)))
    ylim([40 120])
    title('GM')
    subplot(232)
    imagesc(squeeze(WM(:,ex_slice,:)))
    ylim([40 120])
    title('WM')
    subplot(233)
    imagesc(squeeze(CSF(:,ex_slice,:)))
    ylim([40 120])
    title('CSF')
    subplot(234)
    imagesc(squeeze(maskGM_e(:,ex_slice,:)))
    ylim([40 120])
    title('Mask GM')
    subplot(235)
    imagesc(squeeze(maskWM_e(:,ex_slice,:)))
    ylim([40 120])
    title('Mask WM')
    subplot(236)
    imagesc(squeeze(maskCSF_e(:,ex_slice,:)))
    ylim([40 120])
    title('Mask CSF')
% pause
% end


% Mean fMRI signal of WM and CSF

% Extraction using masks
for p=1:size_data(3)
    mask_GM_tmp = maskGM_e(:,:,p);
    mask_WM_tmp = maskWM_e(:,:,p);
    mask_CSF_tmp = maskCSF_e(:,:,p);
    for q=1:size_data(4)
        GM_masked(:,:,p,q) = data(:,:,p,q).*mask_GM_tmp;
        WM_masked(:,:,p,q) = data(:,:,p,q).*mask_WM_tmp;
        CSF_masked(:,:,p,q) = data(:,:,p,q).*mask_CSF_tmp;
    end
end

% Mean computation
for j=1:size_data(4)
    aux1 = WM_masked(:,:,:,j);
    mean_WM(j) = mean2(aux1);

    aux2 = CSF_masked(:,:,:,j);
    mean_CSF(j) = mean2(aux2);    
end

figure,
subplot(211)
plot(mean_WM)
xlim([0 225])
title('Mean fMRI signal of WM')
subplot(212)
plot(mean_CSF)
xlim([0 225])
title('Mean fMRI signal of CSF')

%% 1.c Sum EPI images

% Sum of EPI volumes in the 4th dimension
sumTIME = squeeze(sum(data,4));
sumEPI = squeeze(sum(sumTIME,3)); % After squeezing time is in the 3rd dimension

% Binary mask for the sumEPI image (looking at the histogram to find the
% threshold)
figure(), 
hist(sumEPI,255)
title('histogram of sumEPI image')

mask_sumEPI = sumEPI>0.75e7;
CC = bwconncomp(mask_sumEPI);  % connected components in binary image
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels); % selection of the largest connected component
mask_sumEPI = zeros(size(sumEPI));   
mask_sumEPI(CC.PixelIdxList{idx}) = 1;

% Visualization of the image and the mask
figure, 
subplot(121)
imagesc(imrotate(sumEPI,90))
title('sumEPI image')
subplot(122)
imagesc(imrotate(sumEPI.*mask_sumEPI,90))
title('Masked sumEPI image')

%% 1.d Mask Hammer Atlas

% Load
atlas_raw = load_untouch_nii('C:\Users\Utente\Desktop\S1G1H3\Data\Atlas\Hammers_2_T1_2mm_int.nii')
atlas = double(atlas_raw.img);
% Apply masks
atlas_GM = atlas.*maskGM_e;
atlas_sumEPI = atlas_GM.*mask_sumEPI;

figure, 
imagesc(imrotate(atlas_sumEPI(:,:,ex_slice),90))
title('Example, slice of masked atlas')

%% 1.e ROI time activity curve extraction

% First discard ROIs
discarded_labels = [3, 4, 17, 18, 19, 44, 45, 46, 47, 48, 49, 74, 75]; % from amygdala, cerebellum, brainstem, corpus callosum, substantia nigra, ventricles

for j=1:size(atlas_sumEPI,3)
    temp_slice = squeeze(atlas_sumEPI(:,:,j));
    temp_mask = ones(size(temp_slice)); 
    for i=1:max(temp_slice(:))
        if (find(discarded_labels==i)>0) % ROIs to be discarded
            temp_mask(find(temp_slice==i)) = 0;
        end
        if (size(find(temp_slice==i))<10) % ROIs with less than 10 voxels
            temp_mask(find(temp_slice==i)) = 0;
        end
    end
    atlas_sumEPI_DL(:,:,j) = temp_slice.*temp_mask;
end

figure,
subplot(122)
imagesc(imrotate(atlas_sumEPI_DL(:,:,ex_slice),90))
title('Example, slice of atlas after discarding ROIs')
subplot(121)
imagesc(imrotate(atlas_sumEPI(:,:,ex_slice),90))
title('Example, slice of atlas before discarding ROIs')

% Compute mean fMRI signal
for i=1:max(atlas(:))    
    for j=1:size_data(4)
            temp = data(:,:,:,j);
            ROI = temp(atlas_sumEPI_DL == i);
            roi_fmri_sumEPI(i,j) = mean(ROI(:));
    end
end

% NaNs removal
[roi_fmri_sumEPI,TF] = rmmissing(roi_fmri_sumEPI,1);

figure,
plot(roi_fmri_sumEPI')
xlim([0 225])
title('Mean fMRI signal for each ROI')

%% 2. DATA DENOISING

load('..\Data\FMRI\MOCOparams.mat');

%% 2.a Noise regression

% Regressors matrix
params = [newMOCOparams mean_WM' mean_CSF'];
% Convert regressors to z-scores
zscores = zscore(params);
% Regression
noise_regress_EPI = lscov(zscores,roi_fmri_sumEPI');

% Regression Matrix
reg_mat = zscores*noise_regress_EPI;

% fMRI denoised
fmri_denoised_EPI = roi_fmri_sumEPI' - reg_mat;

figure,
plot(fmri_denoised_EPI)
xlim([0 225])
title('Denoised fMRI')

figure,
imagesc(reg_mat)
title('Regression matrix')

% To save
GC = reg_mat;

%% 2.b Temporal filtering
[n,Wn] = buttord(0.0078,0.002,3,60);
[b,a] = butter(n,Wn,'high');
filt_fmri_EPI = filtfilt(b,a,fmri_denoised_EPI); % We lose the DC component

figure,
subplot(211)
plot(fmri_denoised_EPI)
xlim([0 225])
title('ROIs time activity before temporal filtering')
subplot(212)
plot(filt_fmri_EPI)
xlim([0 225])
title('ROIs time activity after temporal filtering')

% To save
ROIs_fMRI_after_denoising = filt_fmri_EPI;
%% 3. VOLUME CONSORING 

% Find volumes affected by motion artefacts
load('..\Data\FMRI\FDparams.mat');
test_FD= find(FD(:,1)>3.5);
if (length(test_FD)==0)
    disp('No volumes to discard, given the threshold of 3.5mm')
end
% Nothing to discard, after numerical evaluation

%% 4. CHECK OF PREPROCESSING STEP

% Check for right hippocampus region
rx_hip_ROI = 1;
figure
subplot(311)
plot(roi_fmri_sumEPI(rx_hip_ROI,:)), grid on
xlim([0 225])
title('Original time-course of right hippocampus')
subplot(312)
plot(fmri_denoised_EPI(:,rx_hip_ROI)), grid on
xlim([0 225])
title('Denoised')
subplot(313)
plot(filt_fmri_EPI(:,rx_hip_ROI)), grid on
xlim([0 225])
title('Time filtered')
% The drift is removed

%% 5. STATIC FC MATRIX COMPUTATION

% Pairwise Pearson Correlation
[conf_mat_EPI,P_EPI] =  corrcoef(filt_fmri_EPI);
% Functional conectivity matrix
conf_mat_EPI = atanh(conf_mat_EPI);

figure,
imagesc(conf_mat_EPI);
title('FC matrix'); 

% To save
zFC = conf_mat_EPI;

%% 6. MULTIPLE COMPARISON CORRECTION

% FDR (because functional images are spatially correlated)
[h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(P_EPI,0.05,'dep','yes');

%% 7. GRAPH MEASURES

% Compute node degree and node strength

n_roi = size(filt_fmri_EPI,2);

% Binarize confusion matrix (threshold on the p-value associated to the FC
% value)
conf_mat_bin=zeros(n_roi,n_roi);
conf_mat_bin(h==1)= 1;
conf_mat_pval = conf_mat_EPI.*h;

% To save
zFC_corr = conf_mat_pval;

% Node degree
degrees = squeeze(sum(conf_mat_bin,1));
% Node strength
strength = squeeze(sum(conf_mat_pval,1,'omitnan'));
% Betweenness
BC = betweenness_wei(conf_mat_pval);
BC = BC/max(BC);

% To save
DEG = degrees;
STR = strength;
BTW_NORM = BC';

% ROIs with higher metrics values
[val_higher_degrees, higher_degrees]=maxk(degrees,10);
[val_higher_strength, higher_strength]=maxk(strength,10);
[val_higher_BC, higher_BC]=maxk(BC,10);

figure,
subplot(311)
stem(degrees), hold on,
stem(higher_degrees, val_higher_degrees, 'rx')
xlim([0 n_roi])
xticks(sort(higher_degrees))
xtickangle(45)
title('Node degree with 10 highest values')
subplot(312)
stem(strength), hold on,
stem(higher_strength, val_higher_strength, 'rx')
xlim([0 n_roi])
xticks(sort(higher_strength))
xtickangle(45)
title('Node strength with 10 highest values')
subplot(313)
stem(BC), hold on,
stem(higher_BC, val_higher_BC, 'rx')
xlim([0 n_roi])
xticks(sort(higher_BC))
xtickangle(45)
title('Normalized betweenness centrality with 10 highest values')

%% DIFFUSION MRI ANALYSIS

%% DIFFUSION SIGNAL VISUALIZATION AND UNDERSTANDING

% Loading the files
diffusion_volumes = load_untouch_nii('C:\Users\Utente\Desktop\S1G1H3\Data\DMRI\diffusion_volumes.nii');
diffusion_volumes = diffusion_volumes.img;
diffusion_brain_mask = load_untouch_nii('C:\Users\Utente\Desktop\S1G1H3\Data\DMRI\diffusion_brain_mask.nii');
mask = diffusion_brain_mask.img;
bvals = load('C:\Users\Utente\Desktop\S1G1H3\Data\DMRI\bvals');
bvecs = load('C:\Users\Utente\Desktop\S1G1H3\Data\DMRI\bvecs');

% Save some useful variables
nVols = size(diffusion_volumes,4); % The DWIs acquired are nVols = 103 
nSlices = size(diffusion_volumes,3);
nVox = size(diffusion_volumes,1);


%% 1.a Count of DWIs and diffusion shells

% Excluding b = 0
counter = 1;
for i=1:nVols
    if (bvals(i) ~= 0)
        new_bvals(counter) = bvals(i);
        new_bvecs(:,counter) = bvecs(:,i);
        counter = counter + 1;
    end
end

% Number of diffusion shells

% Visual inspection 
% figure(), 
% histogram(new_bvals,size(new_bvals,2)) 
% title('Histogram visualization of bvals')

% Counter
shells_counter = 0;
for j=1:size(new_bvals,2)
    if (shells_counter == 0)
        shells(1) = new_bvals(1);
        shells_counter = shells_counter+1;
    else 
        new_shell = true;
        for k=1:shells_counter
            if (new_bvals(j) > (shells(k)-20) && new_bvals(j) < (shells(k)+20))
                new_shell = false;
            end
        end
        if (new_shell == true)
            shells_counter = shells_counter+1;
            shells(shells_counter) = new_bvals(j);
        end
    end
end

%% 1.b Diffusion signal of voxel populated principally with CSF

% Voxel populated principally with cerebrospinal fluid (we chose a voxel
% for which the mean FA value, computed later, is close to zero)

% FOUND VOXEL: [75, 76, 45] (by visually inspecting the plot of the matrix of
% values of FA for slice 45)
vox = [75,76,45];
diff_signal_unsort = squeeze(diffusion_volumes(vox(1),vox(2),vox(3),:));

% Sorting the signal
[B_sort,I] = sort(bvals);
diff_signal = diff_signal_unsort(I);

figure,
subplot(211)
plot(diff_signal_unsort)
title('Diffusion signal of voxel [75, 76, 45] - Unsorted')
xlim([0 103])
subplot(212)
plot(diff_signal)
xlim([0 103])
title('Diffusion signal of voxel [75, 76, 45] - Sorted')


%% 2. DIFFUSION TENSOR COMPUTATION

%% 2.a Create the new 4D matrix 

% Matrix creation
min_dist = 1000;
min_dist_bval = 0;
for i=1:size(new_bvals,2)
    if (abs(1000-new_bvals(i))<min_dist)
        min_dist = abs(1000-new_bvals(i));
        min_dist_bval = new_bvals(i);
    end
end
    

counter_0 = 1;
counter = 1;
for i=1:nVols
    % Only bvals = 0
    if (bvals(i) == 0)
        new_diffusion_volumes_0(:,:,:,counter_0) = diffusion_volumes(:,:,:,i);
        new_bvals_0(counter_0) = bvals(i);
        new_bvecs_0(:,counter_0) = bvecs(:,i);
        counter_0 = counter_0 + 1;
    end
    % Bvals = 0 and closest to 1000
    if (bvals(i) == 0 || bvals(i) > (min_dist_bval-20) && bvals(i) < (min_dist_bval+20))
        new_diffusion_volumes_min_dist(:,:,:,counter) = diffusion_volumes(:,:,:,i);
        new_bvals_min_dist(counter) = bvals(i);
        new_bvecs_min_dist(:,counter) = bvecs(:,i);
        counter = counter + 1;
    end
end

% New number of volumes 
new_nVols = size(new_diffusion_volumes_min_dist,4);

%% 2b. Using as S0 the voxel-wise value of the first b=0 volume of the available dataset

% Build the B design matrix for the linear least squares approach
for ii=1:length(new_bvals_min_dist)
    B(ii,1)=new_bvecs_min_dist(1,ii)^2;
    B(ii,2)=new_bvecs_min_dist(2,ii)^2;
    B(ii,3)=new_bvecs_min_dist(3,ii)^2;
    B(ii,4)=new_bvecs_min_dist(1,ii)*new_bvecs_min_dist(2,ii);
    B(ii,5)=new_bvecs_min_dist(1,ii)*new_bvecs_min_dist(3,ii);
    B(ii,6)=new_bvecs_min_dist(2,ii)*new_bvecs_min_dist(3,ii);
end
B=new_bvals_min_dist'.*B;

% Normalize the signals and pass to the logarithm for the fit
ind_bvals_zeros = find(bvals==0);
S0=new_diffusion_volumes_0(:,:,:,1);
for ii=1:length(new_bvals_min_dist)
    Slog(:,:,:,ii)=log((new_diffusion_volumes_min_dist(:,:,:,ii)./(S0))+eps);
end

% Initialize the structures which will be used to contain DTI parameters
FA=zeros(nVox,nVox,nSlices);
MD=zeros(nVox,nVox,nSlices);

FirstX=zeros(nVox,nVox,nSlices);
FirstY=zeros(nVox,nVox,nSlices);
FirstZ=zeros(nVox,nVox,nSlices);

% start the cycle to fit the voxel-wise diffusion tensor
for jj=1:nSlices
    
    %print fitting progress
    disp([' Fitting Slice ',num2str(jj)])
    
    for kk=1:1:nVox
        for ll=1:1:nVox
            
            %check if current voxel belongs to the mask
            if (mask(kk,ll,jj) && S0(kk,ll,jj)~=0)
                
                %extract the signal from each voxel
                VoxSignal=squeeze(Slog(kk,ll,jj,:));
                
                %fit the DTI
                D=-inv(B'*B)*B'*VoxSignal;
                
                %reconstruct the diffusion tensor from the fitted
                %parameters
                T=[D(1) D(4)/2 D(5)/2;
                    D(4)/2 D(2) D(6)/2;
                    D(5)/2 D(6)/2 D(3)];
                
                %compute eigenvalues and eigenvectors
                [eigenvects, eigenvals]=eig(T);
                eigenvals=diag(eigenvals);
                
                
                %Manage negative eigenvals
                % if all <0 -> take the absolute value
                %otherwise -> put negatives to zero
                if((eigenvals(1)<0)&&(eigenvals(2)<0)&&(eigenvals(3)<0)), eigenvals=abs(eigenvals);end
                if(eigenvals(1)<0), eigenvals(1)=0; end
                if(eigenvals(2)<0), eigenvals(2)=0; end
                if(eigenvals(3)<0), eigenvals(3)=0; end
                
                First = eigenvects(:,3);
                
                %compute FA
                FAv=(1/sqrt(2))*( sqrt((eigenvals(1)-eigenvals(2)).^2+(eigenvals(2)-eigenvals(3)).^2 + ...
                    (eigenvals(1)-eigenvals(3)).^2)./sqrt(eigenvals(1).^2+eigenvals(2).^2+eigenvals(3).^2) );
                FA(kk,ll,jj)=FAv;
                
                %Compute the MD
                MDv=(eigenvals(1)+eigenvals(2)+eigenvals(3))/3;
                MD(kk,ll,jj)=MDv;
                
                %sort eigenvalues, eigenvectors
                [sorted,idx_sort]=sort(eigenvals);
                
                eigenvals=eigenvals(idx_sort);
                eigenvects=eigenvects(:,idx_sort);
                
                %take principal eigenvector, decompose it
                First = eigenvects(:,3);
                
                FirstX(kk,ll,jj)=abs(First(1))*FAv;
                FirstY(kk,ll,jj)=abs(First(2))*FAv;
                FirstZ(kk,ll,jj)=abs(First(3))*FAv;
                   
            end
        end
    end
end

% Create the colour-encoded directional map
color(:,:,:,1)=FirstX;
color(:,:,:,2)=FirstY;
color(:,:,:,3)=FirstZ;

% Visualize the map
figure,
for ii=1:1:50
    slice=reshape(color(:,:,ii,:),nVox,nVox,3);
    image(imrotate(slice,90))
    title('Map visualization (2b.)')
    pause(0.1)
end
clear slice

FA_2b = FA;
MD_2b = MD;

%% 2c. Using as S0 the voxel-wise mean value of all b=0 volumes of the available dataset

% Build the B design matrix for the linear least squares approach
for ii=1:length(new_bvals_min_dist)
    B(ii,1)=new_bvecs_min_dist(1,ii)^2;
    B(ii,2)=new_bvecs_min_dist(2,ii)^2;
    B(ii,3)=new_bvecs_min_dist(3,ii)^2;
    B(ii,4)=new_bvecs_min_dist(1,ii)*new_bvecs_min_dist(2,ii);
    B(ii,5)=new_bvecs_min_dist(1,ii)*new_bvecs_min_dist(3,ii);
    B(ii,6)=new_bvecs_min_dist(2,ii)*new_bvecs_min_dist(3,ii);
end
B=new_bvals_min_dist'.*B;

% Normalize the signals and pass to the logarithm for the fit
ind_bvals_zeros = find(bvals==0);
S0=mean(new_diffusion_volumes_0(:,:,:,:),4);
for ii=1:length(new_bvals_min_dist)
    Slog(:,:,:,ii)=log((new_diffusion_volumes_min_dist(:,:,:,ii)./(S0))+eps);
end

% Initialize the structures which will be used to contain DTI parameters
FA=zeros(nVox,nVox,nSlices);
MD=zeros(nVox,nVox,nSlices);

FirstX=zeros(nVox,nVox,nSlices);
FirstY=zeros(nVox,nVox,nSlices);
FirstZ=zeros(nVox,nVox,nSlices);



% start the cycle to fit the voxel-wise diffusion tensor
for jj=1:nSlices
    
    %print fitting progress
    disp([' Fitting Slice ',num2str(jj)])
    
    for kk=1:1:nVox
        for ll=1:1:nVox
            
            %check if current voxel belongs to the mask
            if (mask(kk,ll,jj) && S0(kk,ll,jj)~=0)
                
                %extract the signal from each voxel
                VoxSignal=squeeze(Slog(kk,ll,jj,:));
                
                %fit the DTI
                D=-inv(B'*B)*B'*VoxSignal;
                
                %reconstruct the diffusion tensor from the fitted
                %parameters
                T=[D(1) D(4)/2 D(5)/2;
                    D(4)/2 D(2) D(6)/2;
                    D(5)/2 D(6)/2 D(3)];
                
                %compute eigenvalues and eigenvectors
                [eigenvects, eigenvals]=eig(T);
                eigenvals=diag(eigenvals);
                
                
                %Manage negative eigenvals
                % if all <0 -> take the absolute value
                %otherwise -> put negatives to zero
                if((eigenvals(1)<0)&&(eigenvals(2)<0)&&(eigenvals(3)<0)), eigenvals=abs(eigenvals);end
                if(eigenvals(1)<0), eigenvals(1)=0; end
                if(eigenvals(2)<0), eigenvals(2)=0; end
                if(eigenvals(3)<0), eigenvals(3)=0; end
                
                First = eigenvects(:,3);
                
                %compute FA
                FAv=(1/sqrt(2))*( sqrt((eigenvals(1)-eigenvals(2)).^2+(eigenvals(2)-eigenvals(3)).^2 + ...
                    (eigenvals(1)-eigenvals(3)).^2)./sqrt(eigenvals(1).^2+eigenvals(2).^2+eigenvals(3).^2) );
                FA(kk,ll,jj)=FAv;
                
                %Compute the MD
                MDv=(eigenvals(1)+eigenvals(2)+eigenvals(3))/3;
                MD(kk,ll,jj)=MDv;
                
                %sort eigenvalues, eigenvectors
                [sorted,idx_sort]=sort(eigenvals);
                
                eigenvals=eigenvals(idx_sort);
                eigenvects=eigenvects(:,idx_sort);
                
                %take principal eigenvector, decompose it
                First = eigenvects(:,3);
                
                FirstX(kk,ll,jj)=abs(First(1))*FAv;
                FirstY(kk,ll,jj)=abs(First(2))*FAv;
                FirstZ(kk,ll,jj)=abs(First(3))*FAv;
                   
            end
        end
    end
end

% create the colour-encoded directional map
color(:,:,:,1)=FirstX;
color(:,:,:,2)=FirstY;
color(:,:,:,3)=FirstZ;

%visualize the map
figure,
for ii=1:1:50
    slice=reshape(color(:,:,ii,:),nVox,nVox,3);
    image(imrotate(slice,90))
    title('Map visualization (2c.)')
    pause(0.1)
end
clear slice

FA_2c = FA;
MD_2c = MD;

%% 2d. Voxel-wise coefficients of variation

% Total mean of the indexes
m_FA_2b = mean2(FA_2b);
m_MD_2b = mean2(MD_2b);
m_FA_2c = mean2(FA_2c);
m_MD_2c = mean2(MD_2c);

% Computation of CVs
for i=1:size(FA_2c,3)
    for j=1:size(FA_2c,1)
        for k=1:size(FA_2c,2)
            CV_FA_2b(j,k,i) = 100 * (m_FA_2b - FA_2b(j,k,i)) ./ m_FA_2b;
            CV_MD_2b(j,k,i) = 100 * (m_MD_2b - MD_2b(j,k,i)) ./ m_MD_2b;
            CV_FA_2c(j,k,i) = 100 * (m_FA_2c - FA_2c(j,k,i)) ./ m_FA_2c;
            CV_MD_2c(j,k,i) = 100 * (m_MD_2c - MD_2c(j,k,i)) ./ m_MD_2c;
        end
    end
end

% Plot
slice = 72;
colorbar_color = 'jet';
figure,
% for g=1:size(FA_2c,3)
%     slice = g;
    subplot(221)
    imagesc(imrotate((CV_FA_2b(:,:,slice)),90)),colormap(colorbar_color), colorbar
    title('CV FA 2b')
    subplot(222)
    imagesc(imrotate((CV_FA_2c(:,:,slice)),90)),colormap(colorbar_color), colorbar
    title('CV FA 2c')
    subplot(223)
    imagesc(imrotate((CV_MD_2b(:,:,slice)),90)),colormap(colorbar_color), colorbar
    title('CV MD 2b')
    subplot(224)
    imagesc(imrotate((CV_MD_2c(:,:,slice)),90)),colormap(colorbar_color), colorbar
    title('CV MD 2c')
%     pause
% end

% Plot the difference of the mean
figure
subplot(121)
imagesc(imrotate(abs(mean(CV_FA_2b-CV_FA_2c,3)),90)), colormap(colorbar_color), colorbar
title('FA abs difference on mean')
subplot(122)
imagesc(imrotate(abs(mean((CV_MD_2b-CV_MD_2c),3)),90)), colormap(colorbar_color), colorbar
title('MD abs difference on mean')

% Looking at FA and MD difference on the mean, we can say that FA is more
% affected by the different normalization choice, but looking at signle
% slides this may vary


%% 2e. Visualization of the FA and MD maps

% We chose the diffusion tensors fitted using the mean of the volumes with
% b=0, as we think that using the mean the estimation of S0 is more precise
% than using just one single value

% To save
cv_MD = CV_MD_2c;
cv_FA = CV_FA_2c;

% Visualization 
figure,
% for i=1:90
% slice = i;
subplot(121)
imagesc(imrotate(FA_2c(:,:,slice),90), [0 1]),colormap(colorbar_color), colorbar
title('FA 2c')
subplot(122)
imagesc(imrotate(MD_2c(:,:,slice),90), [0 5e-3]),colormap(colorbar_color), colorbar
title('MD 2c')
% pause
% end

% save FA and MD volumes
save_3D_nii(FA_2c, '../Data/DMRI/diffusion_brain_mask.nii', '../Saved_Data/FA.nii');
save_3D_nii(MD_2c, '../Data/DMRI/diffusion_brain_mask.nii', '../Saved_Data/MD.nii');


%% Back to point 1.b:
% Show FA for slice 45 (to select a voxel populated mainly with
% CSF)

figure,
imagesc(FA_2c(:,:,slice)),colormap(colorbar_color), colorbar
title('FA 2c')

% selected voxel (with 'data cursor'): [75 76 45]

%% 2f. Mask the FA and MD maps and extract their mean values in each ROI 

% Mask the maps with GM and sumEPI masks
masked_FA_2c = FA_2c.*maskGM_e;
masked_FA_2c = masked_FA_2c.*mask_sumEPI;

masked_MD_2c = MD_2c.*maskGM_e;
masked_MD_2c = masked_MD_2c.*mask_sumEPI;

slice = 50;
figure,
subplot(121)
imagesc(imrotate(masked_FA_2c(:,:,slice),90), [0 1]), colormap(colorbar_color), colorbar
title('Masked FA 2c')
subplot(122)
imagesc(imrotate(masked_MD_2c(:,:,slice),90), [0 5e-3]), colormap(colorbar_color), colorbar
title('Masked MD 2c')

% Extract mean values in each ROI
for i=1:max(atlas(:))   
    % FA
    ROI = masked_FA_2c(atlas_sumEPI_DL == i);
    roi_FA_mean(i) = mean(ROI);
    % MD
    ROI = masked_MD_2c(atlas_sumEPI_DL == i);
    roi_MD_mean(i) = mean(ROI);
end

% Remove NaNs
[roi_FA_mean,TF] = rmmissing(roi_FA_mean,2);
[roi_MD_mean,TF] = rmmissing(roi_MD_mean,2);

%% DMRI/fMRI INTEGRATION

%% 1. Visual inspection

% Scatterplots
figure,
subplot(231)
scatter(roi_FA_mean, degrees)
title('Node degree VS FA')

subplot(232)
scatter(roi_FA_mean, strength)
title('Node strength VS FA')

subplot(233)
scatter(roi_FA_mean, BC')
title('Node normalized BC VS FA')

subplot(234)
scatter(roi_MD_mean, degrees)
title('Node degree VS MD')

subplot(235)
scatter(roi_MD_mean, strength)
title('Node strength VS MD')

subplot(236)
scatter(roi_MD_mean, BC')
title('Node normalized BC VS MD')

%% 2. Quantitative results

% Pearson's correlation
corr_D_FA = corrcoef(degrees,roi_FA_mean);
corr_S_FA = corrcoef(strength,roi_FA_mean);
corr_BC_FA = corrcoef(BC',roi_FA_mean);
corr_D_MD = corrcoef(degrees,roi_MD_mean);
corr_S_MD = corrcoef(strength,roi_MD_mean);
corr_BC_MD = corrcoef(BC',roi_MD_mean);

%% SAVE REQUIRED VARIABLES
save('../Saved_Data/saved_variables.mat', 'GC','ROIs_fMRI_after_denoising', ...
        'zFC','zFC_corr','DEG','STR','BTW_NORM','cv_MD','cv_FA')
