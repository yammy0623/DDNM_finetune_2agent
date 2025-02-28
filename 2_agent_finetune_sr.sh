EXP="/tmp2/ICML2025/ddnm_finetune"
IMAGE_FOLDER="/tmp2/ICML2025/ddnm_finetune/imagenet"
SEED=232

mkdir -p model/sr_bicubic_imagenet_2_agents_A2C_5
# SR 5
# train ours (1st subtask)
export CUDA_VISIBLE_DEVICES=0
python train.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_ --target_steps 5 --seed $SEED
# eval ours (1st subtask)
python eval.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_eval --target_steps 5 --eval_model_name sr_bicubic_imagenet_2_agents_A2C_5 --subtask1 >> model/sr_bicubic_imagenet_2_agents_A2C_5/2_agents_sub1.txt
# train ours (2nd subtask)
python train.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_ --target_steps 5 --second_stage --seed $SEED
# eval ours
python eval.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_eval --target_steps 5 --eval_model_name sr_bicubic_imagenet_2_agents_A2C_5 >> model/sr_bicubic_imagenet_2_agents_A2C_5/2_agents_sub2.txt

# cp ./model/sr_bicubic_imagenet_2_agents_A2C_5/best.zip ./model/sr_bicubic_imagenet_2_agents_A2C_5/best_orig.zip
# cp ./model/sr_bicubic_imagenet_2_agents_A2C_5/best_2.zip ./model/sr_bicubic_imagenet_2_agents_A2C_5/best_2_orig.zip
# finetune
# for Iter in {1..5}; do
#     echo "Iteration: $Iter"
#     python train.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_ --target_steps 5 --finetune 1 --finetune_ratio 0.2
#     # eval ours (1st subtask)
#     python eval.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_eval --target_steps 5 --eval_model_name sr_bicubic_imagenet_2_agents_A2C_5 --subtask1 >> model/sr_bicubic_imagenet_2_agents_A2C_5/2_agents_fine_${Iter}_sub1.txt
#     # train ours (2nd subtask)
#     python train.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_ --target_steps 5 --second_stage --finetune 1 --finetune_ratio 0.2
#     # eval ours
#     python eval.py --ni --config imagenet_256.yml --exp $EXP --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_5_eval --target_steps 5 --eval_model_name sr_bicubic_imagenet_2_agents_A2C_5 >> model/sr_bicubic_imagenet_2_agents_A2C_5/2_agents_fine_${Iter}_sub2.txt
# done




# SR 10
# # train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_10_ --target_steps 10
# # eval ours (1st subtask
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_10_eval --target_steps 10 --eval_model_name sr_bicubic_imagenet_2_agents_Remain3_A2C_10 --subtask1 >> model/sr_bicubic_imagenet_2_agents_Remain3_A2C_10/subtask1.txt
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_10_ --target_steps 10 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_10_eval --target_steps 10 --eval_model_name sr_bicubic_imagenet_2_agents_Remain3_A2C_10 > model/sr_bicubic_imagenet_2_agents_Remain3_A2C_10/2_agents.txt

# SR 20
# # train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_20_ --target_steps 20
# # eval ours (1st subtask)
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_20_eval --target_steps 20 --eval_model_name sr_bicubic_imagenet_2_agents_Remain3_A2C_20 --subtask1 >> model/sr_bicubic_imagenet_2_agents_Remain3_A2C_20/subtask1.txt
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_20_ --target_steps 20 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "sr_bicubic" --deg_scale 4 --sigma_y 0. -i imagenet_sr_bc_4_20_eval --target_steps 20 --eval_model_name sr_bicubic_imagenet_2_agents_Remain3_A2C_20 > model/sr_bicubic_imagenet_2_agents_Remain3_A2C_20/2_agents.txt

############################################################################################################

# DB 5
# train ours (1st subtask)
# CUDA_VISIBLE_DEVICES=1 python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_5_  --target_steps 5
# # eval ours (1st subtask)
# CUDA_VISIBLE_DEVICES=1 python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_5_eval --target_steps 5 --eval_model_name deblur_gauss_imagenet_2_agents_A2C_5  --subtask1 >> model/deblur_gauss_imagenet_2_agents_A2C_5/subtask1.txt
# # # # train ours (2nd subtask)
# CUDA_VISIBLE_DEVICES=1 python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_5_  --target_steps 5 --second_stage
# # # # eval ours
# CUDA_VISIBLE_DEVICES=1 python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_5_eval --target_steps 5 --eval_model_name deblur_gauss_imagenet_2_agents_A2C_5 >> model/deblur_gauss_imagenet_2_agents_A2C_5/2_agents.txt

# DB 10
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_10_  --target_steps 10
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_10_  --target_steps 10 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_10_eval --target_steps 10 --eval_model_name deblur_gauss_imagenet_2_agents_A2C_10 >> 2_agents.txt

# DB 20
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_20_  --target_steps 20
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_20_  --target_steps 20 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "deblur_gauss" --sigma_y 0. -i imagenet_deblur_g_20_eval --target_steps 20 --eval_model_name deblur_gauss_imagenet_2_agents_A2C_20 >> 2_agents.txt

############################################################################################################

# CL 5
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "colorization" --sigma_y 0. -i imagenet_colorization  --target_steps 5
# # eval ours (1st subtask)
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "colorization" --sigma_y 0. -i imagenet_colorization_eval --target_steps 5 --eval_model_name colorization_imagenet_2_agents_Remain3_A2C_5 --subtask1 >> model/colorization_imagenet_2_agents_Remain3_A2C_5/subtask1.txt
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "colorization" --sigma_y 0. -i imagenet_colorization  --target_steps 5 --second_stage
# # eval ours  
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "colorization" --sigma_y 0. -i imagenet_colorization_eval --target_steps 5 --eval_model_name colorization_imagenet_2_agents_Remain3_A2C_5 >> model/colorization_imagenet_2_agents_Remain3_A2C_5/2_agents.txt

############################################################################################################

# IP 5
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "inpainting" --sigma_y 0. -i imagenet_inpainting --target_steps 5
# eval ours (1st subtask)
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "inpainting" --sigma_y 0. -i imagenet_inpainting_eval --target_steps 5 --eval_model_name inpainting_imagenet_2_agents_A2C_5 --subtask1 >> model/inpainting_imagenet_2_agents_A2C_5/subtask1.txt
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "inpainting" --sigma_y 0. -i imagenet_inpainting --target_steps 5 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "inpainting" --sigma_y 0. -i imagenet_inpainting_eval --target_steps 5 --eval_model_name inpainting_imagenet_2_agents_A2C_5 >> model/inpainting_imagenet_2_agents_A2C_5/2_agents.txt

############################################################################################################
# CS 5
# train ours (1st subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "cs_walshhadamard" --deg_scale 0.25 --sigma_y 0. -i imagenet_cs_wh_025 --target_steps 5
# # eval ours (1st subtask)
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "cs_walshhadamard" --deg_scale 0.25 --sigma_y 0. -i imagenet_cs_wh_025_eval --target_steps 5 --eval_model_name cs_walshhadamard_imagenet_2_agents_Remain3_A2C_5 --subtask1 >> model/cs_walshhadamard_imagenet_2_agents_Remain3_A2C_5/subtask1.txt
# # train ours (2nd subtask)
# python train.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "cs_walshhadamard" --deg_scale 0.25 --sigma_y 0. -i imagenet_cs_wh_025 --target_steps 5 --second_stage
# # eval ours
# python eval.py --ni --config imagenet_256.yml --path_y imagenet --eta 0.85 --deg "cs_walshhadamard" --deg_scale 0.25 --sigma_y 0. -i imagenet_cs_wh_025_eval --target_steps 5 --eval_model_name cs_walshhadamard_imagenet_2_agents_Remain3_A2C_5 >> model/cs_walshhadamard_imagenet_2_agents_Remain3_A2C_5/2_agents.txt

