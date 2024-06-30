data = open('/Volumes/Whale/for MAC/Internship(code)/Github Model/eng_to_hin.txt', 'r', encoding='utf8', errors='ignore').read()
lines = data.split('\n')
total_lines = len(lines)

print("Total number of lines in the dataset:", total_lines)

#Total number of lines in the dataset: 11152