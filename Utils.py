import json 


class GetDocuments:

    # read JSON data from a file
    def read_json_file(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    