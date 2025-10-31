#python 2_Trainer.py --name ChestXRay_10_no_raster --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --val_every 10 --no-raster
#python 2_Trainer.py --name ChestXRay_10_dual_no_raster --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --dual --val_every 10 --no-raster

#python 2_Trainer.py --name ChestXRay_10 --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --val_every 10 
#python 2_Trainer.py --name ChestXRay_10_dual --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --dual --val_every 10 

#python 2_Trainer.py --name ChestXRay_10_NN_no_raster --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --val_every 10 --non-naive --no-raster --resume ../Training/CXRAY_Landmarks_3_10_PreTrain/CXRAY_Landmarks_3_10_PreTrain.pth
#python 2_Trainer.py --name ChestXRay_10_dual_NN_no_raster --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --dual --val_every 10 --non-naive --no-raster --resume ../Training/CXRAY_Landmarks_3_10_PreTrain_Dual/CXRAY_Landmarks_3_10_PreTrain_Dual.pth

#python 2_Trainer.py --name ChestXRay_10_NN --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --val_every 10 --non-naive --resume ../Training/CXRAY_Landmarks_3_10_PreTrain/CXRAY_Landmarks_3_10_PreTrain.pth
#python 2_Trainer.py --name ChestXRay_10_dual_NN --dataset ../Dataset/CXRAY/Landmarks_3_10 --iterations 400000 --dual --val_every 10 --non-naive --resume ../Training/CXRAY_Landmarks_3_10_PreTrain_Dual/CXRAY_Landmarks_3_10_PreTrain_Dual.pth

#python 2_Trainer.py --name CAMUS_NN_no_raster --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --non-naive --resume ../Training/CAMUS_Landmarks_3_10_PreTrain/CAMUS_Landmarks_3_10_PreTrain.pth --no-raster
#python 2_Trainer.py --name CAMUS_NN_dual_no_raster --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --non-naive --dual --resume ../Training/CAMUS_Landmarks_3_10_PreTrain_Dual/CAMUS_Landmarks_3_10_PreTrain_Dual.pth --no-raster

#python 2_Trainer.py --name CAMUS_NN --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --non-naive --resume ../Training/CAMUS_Landmarks_3_10_PreTrain/CAMUS_Landmarks_3_10_PreTrain.pth 
#python 2_Trainer.py --name CAMUS_NN_dual --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --non-naive --dual --resume ../Training/CAMUS_Landmarks_3_10_PreTrain_Dual/CAMUS_Landmarks_3_10_PreTrain_Dual.pth 

#python 2_Trainer.py --name CAMUS_s_nr --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --no-raster --pretrain
#python 2_Trainer.py --name CAMUS_dual_nr --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --dual --no-raster --pretrain

#python 2_Trainer.py --name CAMUS_sp --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --pretrain
#python 2_Trainer.py --name CAMUS_dualp --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 500000 --dual --pretrain

#python 2_Trainer.py --name CAMUS_atlas_seg_NN --dataset ../Dataset/CAMUS/Landmarks_3_10 --iterations 300000 --non-naive --raster-input --resume ../Training/CAMUS_Landmarks_3_10_PreTrainFromMask/CAMUS_Landmarks_3_10_PreTrainFromMask.pth

#python 8_Trainer_MSE.py --name CAMUS_HybridGNet --dataset ../Dataset/CAMUS/Landmarks_3_10 --epochs 30 --warm_up_it 100000000000
#python 8_Trainer_MSE.py --name CAMUS_HybridGNet_Raster --dataset ../Dataset/CAMUS/Landmarks_3_10 --epochs 30 

#python 2_Trainer.py --name HC18 --dataset ../Dataset/HC18/Landmarks_3_10 --dual --iterations 300000 
#python 2_Trainer.py --name JNU-IFM --dataset ../Dataset/JNU-IFM/Landmarks_3_10 --dual --iterations 300000 
#python 2_Trainer.py --name Merged --dataset ../Dataset/Merged/Landmarks_3_10 --dual --iterations 600000 
#python 2_Trainer.py --name PSFHS --dataset ../Dataset/PSFHS/Landmarks_3_10 --dual --iterations 300000 

#python 2_Trainer.py --name PRF_no_raster --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --no-raster --pretrain
#python 2_Trainer.py --name PRF_dual_no_raster --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --dual --no-raster --pretrain 

#python 2_Trainer.py --name PRF_no_raster_NN_v2 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --no-raster --pretrain --non-naive --val_every 10 --elasticity_w 1000 --curvature_w 1000
#python 2_Trainer.py --name PRF_dual_no_raster_NN --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --dual --no-raster --pretrain --non-naive --val_every 10 --elasticity_w 1000 --curvature_w 1000

#python 2_Trainer.py --name PRF_no_raster_v4_NN --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --non-naive --pretrain --val_every 10 --elasticity_w 2000 --curvature_w 1000 --edge_w 3 --no-raster
#python 2_Trainer.py --name PRF_dual_no_raster_v4_NN --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --non-naive --dual --pretrain --val_every 10 --elasticity_w 2000 --curvature_w 1000 --edge_w 3 --no-raster
#python 2_Trainer.py --name PRF_no_raster_v4 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --pretrain --val_every 10 --elasticity_w 2000 --curvature_w 1000 --edge_w 2 --no-raster
#python 2_Trainer.py --name PRF_dual_no_raster_v4 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --dual --pretrain --val_every 10 --elasticity_w 2000 --curvature_w 1000 --edge_w 3 --no-raster

#python 2_Trainer.py --name PRF_no_raster_v5 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --pretrain --val_every 10 --elasticity_w 1500 --curvature_w 1000 --edge_w 2 --no-raster --resume  ../Training/PRF_no_raster_v4/PRF_no_raster_v4.pth
#python 2_Trainer.py --name PRF_dual_no_raster_v5 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 500000 --dual --pretrain --val_every 10 --elasticity_w 1500 --curvature_w 1000 --edge_w 3 --no-raster --resume ../Training/PRF_dual_no_raster_v4/PRF_dual_no_raster_v4.pth

#python 2_Trainer.py --name PRF_no_raster_NN --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 600000 --pretrain --val_every 10 --elasticity_w 2000 --curvature_w 1500 --edge_w 2 --no-raster --non-naive
#python 2_Trainer.py --name PRF_no_raster_NN_v2 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 600000 --pretrain --val_every 10 --elasticity_w 1500 --curvature_w 1000 --edge_w 2 --no-raster --non-naive --resume ../Training/PRF_no_raster_NN/PRF_no_raster_NN.pth

#python 2_Trainer.py --name PRF_dual_no_raster_NN --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 600000 --dual --pretrain --val_every 10 --elasticity_w 2000 --curvature_w 1500 --edge_w 2 --no-raster --non-naive
#python 2_Trainer.py --name PRF_dual_no_raster_NN_v2 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 600000 --dual --pretrain --val_every 10 --elasticity_w 1500 --curvature_w 1000 --edge_w 2 --no-raster --non-naive --resume ../Training/PRF_dual_no_raster_NN/PRF_dual_no_raster_NN.pth

python 2_Trainer.py --name PRF_dual_NN_prod --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 800000 --dual --prod --pretrain --val_every 10 --elasticity_w 2000 --curvature_w 1500 --edge_w 2 --no-raster --non-naive
python 2_Trainer.py --name PRF_dual_NN_prod_v2 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 400000 --dual --prod --pretrain --val_every 10 --elasticity_w 1500 --curvature_w 1000 --edge_w 2 --no-raster --non-naive --resume ../Training/PRF_dual_NN_prod/PRF_dual_NN_prod.pth
python 2_Trainer.py --name PRF_dual_NN_prod_v3 --dataset ../Dataset/PAXRay_Front/Landmarks_3_10 --iterations 400000 --dual --prod --pretrain --val_every 10 --elasticity_w 1500 --curvature_w 1000 --edge_w 2 --no-raster --non-naive --resume ../Training/PRF_dual_NN_prod_v2/PRF_dual_NN_prod_v2.pth