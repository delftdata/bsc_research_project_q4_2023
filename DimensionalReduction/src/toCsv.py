import csv

def create_csv_file(dr_method, classification_methods, data):
    print(classification_methods)
    filename = (str(dr_method) + "_" + str(classification_methods) + ".csv")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['var value', 'accuracy', 'number of features'])
        for entry in data:
            var_value = entry['var value']
            accuracy = entry['accuracy']
            num_features = entry['number of features']
            writer.writerow([var_value, accuracy, num_features])

def parse_text_file(filename):
    current_dr_method = None
    current_classification_methods = None
    current_data = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('dr method:'):
                current_dr_method = line.split(':')[1].strip().split('.')[-1].strip().split(" ")[0]
                if str(line.split("|")[1].split(":  ")[1]).startswith('S'):
                    current_classification_methods = "SVM"
                else:
                    current_classification_methods = line.split('|')[1].split(':  {\'')[-1].split('\'')[0]
                print(current_dr_method)
                print(current_classification_methods)
            elif line.startswith('var value:'):
                var_value = float(line.split(':')[1].strip().split('|')[0])
                accuracy = float(line.split(':')[2].strip().split('|')[0])
                num_features = int(line.split(':')[3].strip())
                current_data.append({
                    'var value': var_value,
                    'accuracy': accuracy,
                    'number of features': num_features
                })
            elif line.strip() == '':
                if current_dr_method and current_classification_methods and current_data:
                    create_csv_file(current_dr_method, current_classification_methods, current_data)
                    current_data = []

# Provide the path to your text file


if __name__ == '__main__':
    text_file_path = 'results.txt'
    parse_text_file(text_file_path)




