import pandas as pd

def csv_to_txt(input_csv, output_txt):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    # Save as text file with space-separated values and each row in a new line
    df.to_csv(output_txt, sep=' ', index=False)

# Example usage:
csv_to_txt(r"abveninput.csv", "abveninput.txt")