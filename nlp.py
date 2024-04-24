import csv

csv_path = '/Users/jabezagyemang-prempeh/Desktop/NLP/test_subset.csv'
df = pd.read_csv(csv_path)
image_root_dir = '/Users/jabezagyemang-prempeh/Desktop/NLP/test'
mapping_dict = {}  # Key: Test ID, Value: Image Path
for index, row in df.iterrows():
    test_id = row['study_id'] 
    mapping_dict[str(int(test_id))] = None

def find_image(id):
    for dirpath, dirnames, filenames in os.walk(image_root_dir):
        for dir in dirnames:
            if id == dir[1:]:
                return dirpath + "/" + id
    return "NOT FOUND"
        
for id in mapping_dict:
    mapping_dict[id] = find_image(id)

with open("id_to_dir.csv","w") as file:
    csv_lst = [["ID","LOCAL DIRECTORY"]]
    for id,directory in mapping_dict.items():
        temp = [id,directory]
        csv_lst.append(temp)

    csv.writer(file).writerows(csv_lst)

#preprocessing test_subset
#creating dic for id_to_dir.csv

id_to_path = {}
with open("all_files/id_to_dir.csv", "r") as file:
    count = 0
    for row in csv.reader(file):
        if count == 0:
            count += 1
            continue
        # print(row)
        id_to_path[row[0]] = "all_files/"+"/".join(row[-1].split('/')[6:-1] + [f"s{row[-1].split('/')[-1]}"])

#adding column to i
#and writing to new file
with open("all_files/test_subset.csv","r+") as second_file:
    count = 0
    new_file = []
    for row in csv.reader(second_file):
        if count == 0: 
            row.append("path")
            count += 1
            new_file.append(row)
        else:
            row.append(id_to_path[row[0]])
            for ind,element in enumerate(row):
                if element == '':
                    row[ind] = '0.0'
                # print(row)
            if row[-1] == 'all_files/sNOT FOUND':
                #skipping row
                print(f"skipping row = {row}")
                continue
            new_file.append(row)



# name of csv file
filename = "new_file.csv"
 
# writing to csv file
with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(new_file[0])
    csvwriter.writerows(new_file[1:])


        