import json

import routes as R

# ontology
with open(R.ontology_path, 'r') as f:
    ontology_data = json.load(f)

id_to_name = {x["id"]: x["name"] for x in ontology_data}
name_to_id = {x["name"]: x["id"] for x in ontology_data}

def get_object_by_id(data, id_value):
    for entry in data:
        if entry['id'] == id_value:
            return entry
    return None

music = get_object_by_id(ontology_data, name_to_id["Music"])
task_ids = music["child_ids"]
tasks = [ get_object_by_id(ontology_data, id) for id in task_ids ]
tasks_labels = {}
for task in tasks:
    tasks_labels[task["id"]] = task["child_ids"]

if __name__=="__main__":
    for task in tasks:
        print(task["name"])
        for id in task["child_ids"]:
            print("\t", id_to_name[id])