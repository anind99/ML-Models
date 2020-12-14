import csv, os

def load_question_meta(path='/data'):
	path = os.path.join(path, "question_meta.csv")
	if not os.path.exists(path):
		raise Exception("The specified path {} does not exist.".format(path))
	
	# Initialize the data.
	metadata = {}
	# Iterate over the row to fill in the data.
	with open(path, "r") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			try:
				metadata[int(row[0])] = [int(s) for s in row[1][1:-1].split(', ')]
			except ValueError:
				# Pass first row.
				pass
	return metadata
