import glob
import os
import subprocess
import zipfile
notebook_file = 'HW3.ipynb'
out_file = notebook_file.split('.')[0] + '_non_programming'
code_files = ['**.py']
patterns_to_ignore = ['**/__pycache__', '**/__pycache__/**']
submission_zip = 'HW3_programming.zip'


def convert_notebook_to_html(notebook_file):
    command = (
        f'jupyter nbconvert {notebook_file} --to webpdf --output {out_file}')
    subprocess.run(command, shell=True, check=True)
    print(f'Successfully converted notebook to PDF')


def zip_folder(folder_path, zipf: zipfile.ZipFile):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, os.path.dirname(folder_path))
            zipf.write(full_path, arcname)


def create_zip_archive(patterns_to_zip: list[str], zip_filename: str,
    patterns_to_ignore=None):
    with zipfile.ZipFile(zip_filename, 'w') as zip_file:
        for pattern in patterns_to_zip:
            filepaths = glob.glob(pattern, recursive=True)
            if not filepaths:
                print(f'Warning: No files found matching pattern: {pattern}')
                continue
            for filepath in filepaths:
                if patterns_to_ignore:
                    if any(glob.fnmatch.fnmatch(filepath, ignore_pattern) for
                        ignore_pattern in patterns_to_ignore):
                        print(f'Ignored {filepath}')
                        continue
                zip_file.write(filepath, os.path.relpath(filepath))
                print(f'Added {filepath} to the zip archive')


command = ' playwright install --with-deps --only-shell chromium'
subprocess.run(command, shell=True, check=True)
convert_notebook_to_html(notebook_file)
create_zip_archive(code_files, submission_zip, patterns_to_ignore)
print('\n\nDone!')
print(
    f'Please submit `{submission_zip}` to the programming assignment and `{out_file}.pdf` to the non-programming assignment on Gradescope'
    )
print("Don't forget to assign pages correctly!!")
