
def write_predicted_blomap_file(predicted_contact_map, output_filepath):
    f = open(output_filepath, 'w+')
    content = predicted_contact_map.to_csv(None, header=False, index=False).split('\n')
    for line in content:
        f.write(line + '\n')
