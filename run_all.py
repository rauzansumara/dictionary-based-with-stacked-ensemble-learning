import os
import pandas as pd
import classify

def run():
    
    # get dataset's names from 'datasets' folder
    subfolders = [os.path.split(f.path)[-1] for f in os.scandir('datasets') if f.is_dir()]
    # subfolders = ['MiddlePhalanxOutlineCorrect','ProximalPhalanxOutlineCorrect']

    # looping clasification
    results = []
    for name in subfolders:
        try:
            tr = classify.run(name)
            results.extend(tr)
        except:
            print(f'Error is found, the {name} dataset is unable to process')
    
    # save evaluation result
    pd.DataFrame(results).to_excel(f'results/output_all.xlsx', index=False)

if __name__ == "__main__":
    run()