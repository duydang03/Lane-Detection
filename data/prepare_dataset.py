from Utils import extract_lane_masks

json_dir = "path_to_train_json"
json_dir_val = "path_to_val_json"

output_mask_dir = "path_to_train_masks"
output_mask_dir_val = "path_to_val_masks"

extract_lane_masks(json_dir, output_mask_dir, image_size=(720, 1280))
extract_lane_masks(json_dir_val, output_mask_dir_val, image_size=(720, 1280))

print("Mask generation completed.")