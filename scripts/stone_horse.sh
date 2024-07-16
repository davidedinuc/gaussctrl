ns-train splatfacto --output-dir unedited_models --experiment-name stone_horse --viewer.quit-on-train-completion True nerfstudio-data --data data/stone_horse

ns-train gaussctrl --load-checkpoint unedited_models/stone_horse/splatfacto/2024-07-11_173710/nerfstudio_models/step-000029999.ckpt --experiment-name stone_horse --output-dir outputs --pipeline.datamanager.data data/stone_horse --pipeline.prompt "a photo of a giraffe in front of the museum" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'stone horse' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/stone_horse/splatfacto/2024-07-11_173710/nerfstudio_models/step-000029999.ckpt --experiment-name stone_horse --output-dir outputs --pipeline.datamanager.data data/stone_horse --pipeline.prompt "a photo of a zebra in front of the museum" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'stone horse' --viewer.quit-on-train-completion True 
