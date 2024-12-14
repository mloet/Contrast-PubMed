import json
import argparse
from pathlib import Path

def find_and_write_unique_lines(file1_path: str, file2_path: str, output_path: str) -> None:
    """
    Finds lines from file1 that don't exist in file2 and writes them to a new JSONL file.
    
    Args:
        file1_path (str): Path to first JSONL file
        file2_path (str): Path to second JSONL file
        output_path (str): Path where the output JSONL file should be written
        
    Returns:
        None
    """
    # Read both files into lists of dictionaries
    file1_lines = []
    file2_lines = []
    
    with open(file1_path, 'r', encoding='utf-8') as f1:
        for line in f1:
            try:
                file1_lines.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
                
    with open(file2_path, 'r', encoding='utf-8') as f2:
        for line in f2:
            try:
                file2_lines.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    unique_lines = []
    for line in file1_lines:
        if line not in file2_lines:
            unique_lines.append(line)
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in unique_lines:
            json.dump(line, f_out, ensure_ascii=False)
            f_out.write('\n')
    
    print(f"Found {len(unique_lines)} unique lines and wrote them to {output_path}")

def main():
    print(1)
    parser = argparse.ArgumentParser(
        description='Find lines in first JSONL file that do not exist in second JSONL file'
    )
    parser.add_argument(
        '--file1',
        type=str,
        help='Path to first JSONL file'
    )
    parser.add_argument(
        '--file2',
        type=str,
        help='Path to second JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output JSONL file'
    )
    
    args = parser.parse_args()
    
    find_and_write_unique_lines(args.file1, args.file2, args.output)

if __name__ == '__main__':
    main()