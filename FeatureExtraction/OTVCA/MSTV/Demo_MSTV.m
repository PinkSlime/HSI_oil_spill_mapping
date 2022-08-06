clear all
close all
for oil_index=1:2

        
        load (['dwh_',num2str(oil_index),'.mat'       ]);
        
        load (['GT_',num2str(oil_index),'.mat'       ]);
        
        Tr=imread(['Tr_',num2str(oil_index),'.tif'  ]);
        
        
        Te=GT-Tr;
        GT=uint8(GT);
        Tr=uint8(Tr);
        Te=uint8(Te);
        %
       
        %     load(['Te_' ,num2str(sample_ratio,'%.2f') ,'.mat'       ]);
        
        %time
        T = 0;
        tic;
        %% compute the EMAP features
        %%
         [no_lines, no_rows, no_bands] = size(img);  
        [ OA,AA,K,PA ,result ] = wzh_MSTV(img,GT,Tr,Te);
        T = T + toc;
        %     PA = acc_Mean(1:dim,1);
        %     OA = acc_Mean(dim+2,1);
        %     K = acc_Mean(dim+3,1);
        %     AA = mean(PA);
        
        %%
        CM=uint8(result );
%        CM=uint8(CM );

        
       
         save ( [ 'MSTV_',   num2str(oil_index), '.mat'] , 'CM'  )   
                
                
        color_map=[
            0,0,0;
            95,69,255;
            
            172,165,199;
            255,243,219;
            
            ];
        color_map=color_map/255;
        %         out_put=[AA,OA,K,PA',NaN,NaN,NaN,NaN,NaN,NaN,T];
        %         xlswrite('D:\oil\new_oil\Result\Statistics.xlsx',out_put,  num2str(sample_ratio,'%.2f'),['B',num2str(59+oil_index-1)]  )
        %         imwrite(CM ,color_map, ['D:\oil\new_oil\Result\PCA\oil_', num2str(oil_index), '_' ,num2str(sample_ratio,'%.2f'), '.png']);
        %         xlswrite('D:\oil\new_oil\Result\Statistics.xlsx',out_put, num2str(sample_ratio,'%.2f') , ['B',num2str(66)]  )
        %         imwrite(CM ,color_map, ['D:\oil\new_oil\Result\PCA\ploil_' ,num2str(sample_ratio,'%.2f'),   '.png']);
        gt=reshape(GT,1,no_lines*no_rows);
        cm=reshape(CM,1,no_lines*no_rows);

        [aa oa ua pa K confu]=new_confusion(gt,cm);
        out_put=[aa,oa,K,pa'];
        xlswrite('D:\oil\add\DWH\re\MSTV.xlsx',  out_put,  'Sheet1' , ['B',num2str(oil_index)] )
        imwrite(CM ,color_map, ['D:\oil\add\DWH\re\MSTV_',   num2str(oil_index),  '.png']);
    end
