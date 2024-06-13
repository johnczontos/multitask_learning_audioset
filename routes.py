import os

# routes
dataset_script_path = "/nfs/guille/eecs_research/soundbendor/datasets/audio_tagging/AudioSet/AudioSet.py"
data_dir = "/nfs/guille/eecs_research/soundbendor/datasets/audio_tagging/AudioSet/data"
ontology_path = os.path.join(data_dir, "ontology.json")
hf_cache_dir = '/nfs/guille/eecs_research/soundbendor/datasets/audio_tagging/AudioSet/cache'
pt_cache_dir = '/nfs/guille/eecs_research/soundbendor/datasets/audio_tagging/AudioSet/pt_cache'
train_cache_dir = os.path.join(pt_cache_dir, "train")
valid_cache_dir = os.path.join(pt_cache_dir, "valid")