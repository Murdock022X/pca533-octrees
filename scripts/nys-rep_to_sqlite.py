import os
rootdir = os.path.join("..", "outputs_031126")

for root, subFolders, files in os.walk(rootdir):
    for file in files:
        if(file.endswith(".sqlite")):
            os.remove(os.path.join(root,file))
        if(file.endswith("nsys-rep")):
            base_name, old_extension = os.path.splitext(file)
            command = "nsys export -t sqlite --include-json true "+ os.path.join(root,file) + " -o " + os.path.join(root, base_name + ".sqlite")
            #print(command)
            os.system(command)