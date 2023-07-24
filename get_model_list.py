import json
import timm

timm_model_list = timm.list_models()
print("timm_model_list: ", timm_model_list)
print("len(timm_model_list): ", len(timm_model_list))
with open("timm_model_list.json", "w") as json_file:
    json.dump(timm_model_list, json_file)
