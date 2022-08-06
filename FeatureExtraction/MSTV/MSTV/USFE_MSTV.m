function [acc_Mean , CM] = USFE_MSTV(HSI, Tr, Te, dim, Trees)
[FE_MSTV]= MSTV_Xu(HSI, dim);
[acc_Mean,acc_std,CM]=RF_ntimes_overal(FE_MSTV,Tr,Te,Trees);
