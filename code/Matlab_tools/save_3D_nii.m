function save_3D_nii(mat,maskfile,niiName)

%This function takes as input the parameter matrices (e.g FA or MD) and
%saves them as a new volume
%  INPUTS:
%  
%     -mat: matrix containing the voxel wise parameters
%     -maskfile: name (complete with path) of the file containing the brain mask
%     -niiName: name (complete with path) of the newly created file

%create the new Nii
newNii=create_3D_nii(maskfile,mat);

%save the new Nii
save_untouch_nii(newNii,niiName);